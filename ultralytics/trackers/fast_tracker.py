# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from .basetrack import TrackState
from .byte_tracker import BYTETracker, STrack
from .utils import matching
from .utils.stracks import merge_track_pools


class FastSTrack(STrack):
    """Single-object track for FastTracker with occlusion-aware state.

    Extends :class:`STrack` with a bounded ring-buffer of recent Kalman means and per-track occlusion
    bookkeeping. The history buffer enables rolling the Kalman state back to a pre-occlusion frame
    when a neighbor suddenly covers the target. The buffer is a fixed-size :class:`collections.deque`, so memory stays
    bounded regardless of track lifetime.

    Attributes:
        mean_history (collections.deque): Bounded ring-buffer of recent Kalman mean vectors, newest last, capped at
            ``history_len`` entries.
        not_matched (int): Consecutive frames this track has failed to match a detection.
        is_occluded (bool): True while the track is hidden behind another target.
        occluded_len (int): Consecutive frames the track has been continuously occluded.
        last_occluded_frame (int): Frame id when occlusion was last detected, or -1 if never occluded.
        was_recently_occluded (bool): Sticky flag kept for ``occ_reappear_window`` frames, used by :class:`FastTracker`
            to extend the re-find window for tracks that went lost while occluded.

    Examples:
        >>> from ultralytics.trackers.utils.kalman_filter import KalmanFilterXYAH
        >>> t = FastSTrack([100, 200, 50, 80, 0], score=0.9, cls=0, history_len=8)
        >>> t.activate(KalmanFilterXYAH(), frame_id=1)
        >>> len(t.mean_history)
        1
    """

    def __init__(self, xywh: list[float], score: float, cls: Any, history_len: int = 16):
        """Initialize a FastSTrack.

        Args:
            xywh (list[float]): Bounding box in ``(x, y, w, h, idx)`` or ``(x, y, w, h, angle, idx)`` format.
            score (float): Detection confidence in [0, 1].
            cls (Any): Class label for the detection.
            history_len (int): Maximum number of past Kalman mean vectors kept for occlusion rollback.
        """
        super().__init__(xywh, score, cls)
        self.mean_history: deque = deque(maxlen=history_len)
        self.not_matched = 0
        self.is_occluded = False
        self.occluded_len = 0
        self.last_occluded_frame = -1
        self.was_recently_occluded = False

    def _push_history(self):
        """Append a copy of the current Kalman mean to the bounded history buffer."""
        if self.mean is not None:
            self.mean_history.append(self.mean.copy())

    def activate(self, kalman_filter, frame_id: int):
        """Activate the track and seed its mean history.

        Args:
            kalman_filter (KalmanFilterXYAH): Shared Kalman filter instance.
            frame_id (int): Frame id at which the track is created.
        """
        super().activate(kalman_filter, frame_id)
        self._push_history()

    def re_activate(self, new_track, frame_id: int, new_id: bool = False):
        """Re-activate a previously lost track.

        Args:
            new_track (FastSTrack): Detection used to revive this track.
            frame_id (int): Current frame id.
            new_id (bool): If True, assign a fresh track id instead of reusing the old one.
        """
        super().re_activate(new_track, frame_id, new_id=new_id)
        self._push_history()

    def update(self, new_track, frame_id: int):
        """Update the track with a newly matched detection.

        Args:
            new_track (FastSTrack): Matched detection for this frame.
            frame_id (int): Current frame id.
        """
        super().update(new_track, frame_id)
        self._push_history()


class FASTTracker(BYTETracker):
    """Occlusion-aware ByteTrack-style multi-object tracker.

    Adapted from the reference implementation in the FastTracker paper (arXiv:2508.14370). FastTracker extends
    :class:`BYTETracker` with lightweight mechanisms that reduce ID switches through crowd occlusions without
    sacrificing throughput. Unmatched tracks whose area is strongly covered by an active neighbor are flagged as
    occluded and their Kalman state is rolled back to a pre-occlusion frame, with a one-shot bbox enlargement and
    dampened motion so they survive the occlusion. An occluded track is kept alive for an extra grace window before
    being marked lost, and once lost it stays re-findable for an extended window beyond the regular ``track_buffer``.
    New detections that strongly overlap an already-active track are suppressed at spawn time to prevent ghost IDs.

    All added work uses vectorized IoU / coverage matrices and only runs on unmatched tracks, so the per-frame overhead
    over :class:`BYTETracker` stays on the order of a few hundred microseconds.

    Attributes:
        reset_velocity_offset_occ (int): Number of frames to look back when restoring Kalman velocity at occlusion
            onset.
        reset_pos_offset_occ (int): Number of frames to look back when restoring Kalman position at occlusion onset.
        enlarge_bbox_occ (float): One-shot multiplier applied to the bbox height when occlusion is first detected.
        dampen_motion_occ (float): Multiplier in [0, 1] applied to Kalman velocity during occlusion.
        active_occ_to_lost_thresh (int): Maximum consecutive occluded frames before a track is marked lost anyway.
        init_iou_suppress (float): IoU threshold above which a new detection is prevented from spawning a fresh track.
            Set to 1.0 to disable suppression.
        occ_cover_thresh (float): Fraction of a track's area that must be covered by another active track to declare
            occlusion.
        occ_reappear_window (int): Frames a recently-occluded lost track stays re-findable beyond the regular
            ``track_buffer``.

    Methods:
        update: Consume a frame's detections and return the currently-tracked objects.
        init_track: Build :class:`FastSTrack` instances from a ``Results``-like object.

    Examples:
        Plug FastTracker into a YOLO model via the bundled config:
        >>> from ultralytics import YOLO
        >>> model = YOLO("yolo26s.pt")
        >>> model.track("video.mp4", tracker="fasttrack.yaml")

        Drive FastTracker directly with your own detections:
        >>> from ultralytics.trackers import FastTracker
        >>> from ultralytics.utils import YAML, IterableSimpleNamespace
        >>> cfg = IterableSimpleNamespace(**YAML.load("ultralytics/cfg/trackers/fasttrack.yaml"))
        >>> tracker = FastTracker(cfg, frame_rate=30)
        >>> tracks = tracker.update(detections)
    """

    def __init__(self, args, frame_rate: int = 30):
        """Initialize FastTracker with tunables read from ``args``.

        Any FastTracker-specific key missing on ``args`` falls back to a sensible default, so
        FastTracker can also be driven by a plain ByteTrack config.

        Args:
            args (Namespace | IterableSimpleNamespace): Parsed tracker config. Must provide the BYTETracker keys
                (``track_high_thresh``, ``track_low_thresh``, ``new_track_thresh``, ``track_buffer``, ``match_thresh``,
                ``fuse_score``) and may provide the FastTracker-specific keys described in the class docstring.
            frame_rate (int): Source video frame rate.
        """
        super().__init__(args, frame_rate=frame_rate)
        # Occlusion-handling knobs (fall back to sensible defaults if absent on args)
        self.reset_velocity_offset_occ = int(getattr(args, "reset_velocity_offset_occ", 5))
        self.reset_pos_offset_occ = int(getattr(args, "reset_pos_offset_occ", 3))
        self.enlarge_bbox_occ = float(getattr(args, "enlarge_bbox_occ", 1.1))
        self.dampen_motion_occ = float(getattr(args, "dampen_motion_occ", 0.5))
        self.active_occ_to_lost_thresh = int(getattr(args, "active_occ_to_lost_thresh", 10))
        self.init_iou_suppress = float(getattr(args, "init_iou_suppress", 0.7))
        self.occ_cover_thresh = float(getattr(args, "occ_cover_thresh", 0.7))
        self.occ_reappear_window = int(getattr(args, "occ_reappear_window", 40))
        # Cap history to the max rollback we'll need + small slack.
        self._history_len = max(self.reset_velocity_offset_occ, self.reset_pos_offset_occ) + 4

    def init_track(self, results, img: np.ndarray | None = None) -> list[FastSTrack]:
        """Build :class:`FastSTrack` instances from a ``Results``-like object.

        Args:
            results (Any): Object exposing ``xywh`` (or ``xywhr``), ``conf``, and ``cls``.
            img (np.ndarray | None): Current BGR frame. Unused by FastTracker; accepted for signature parity with other
                trackers.

        Returns:
            (list[FastSTrack]): One :class:`FastSTrack` per detection, empty if no detections.
        """
        if len(results) == 0:
            return []
        bboxes = results.xywhr if hasattr(results, "xywhr") else results.xywh
        bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)
        hl = self._history_len
        return [FastSTrack(xywh, s, c, history_len=hl) for (xywh, s, c) in zip(bboxes, results.conf, results.cls)]

    def update(self, results, img: np.ndarray | None = None, feats: np.ndarray | None = None) -> np.ndarray:
        """Advance the tracker by one frame and return the currently active tracked objects.

        Performs the BYTE two-pass association (high-score then low-score), then runs occlusion
        handling on any tracks that remain unmatched, associates unconfirmed tracks against leftover
        detections, spawns new tracks (subject to IoU suppression), and retires lost tracks with a
        grace window for recently-occluded ones.

        Args:
            results (Any): ``Results``-like object exposing ``xywh`` (or ``xywhr``), ``conf``, and ``cls``, and supporting
                boolean / ndarray indexing.
            img (np.ndarray | None): Current frame. Unused by FastTracker.
            feats (np.ndarray | None): Optional per-detection appearance features. Unused by FastTracker; accepted for
                signature compatibility.

        Returns:
            (np.ndarray): Float32 array with one row per activated track of the form ``[..., track_id, score, cls,
                det_idx]``. Leading coordinates are ``xyxy`` for standard boxes or ``xywha`` for oriented boxes.
        """
        self.frame_id += 1
        activated_stracks: list[FastSTrack] = []
        refind_stracks: list[FastSTrack] = []
        lost_stracks: list[FastSTrack] = []
        removed_stracks: list[FastSTrack] = []

        scores = results.conf
        remain_inds = scores >= self.args.track_high_thresh
        inds_low = scores > self.args.track_low_thresh
        inds_high = scores < self.args.track_high_thresh
        inds_second = inds_low & inds_high

        results_second = results[inds_second]
        results_keep = results[remain_inds]

        detections = self.init_track(results_keep, img)

        unconfirmed: list[FastSTrack] = []
        tracked_stracks: list[FastSTrack] = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        # --- First association (high-score detections) ---
        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)
        self.multi_predict(strack_pool)
        dists = self.get_dists(strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        self._apply_matches(matches, strack_pool, detections, activated_stracks, refind_stracks)

        # --- Second association (low-score detections) ---
        detections_second = self.init_track(results_second, img)
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        if self.args.fuse_score:
            dists = matching.fuse_score(dists, detections_second)
        matches, u_track, _ = matching.linear_assignment(dists, thresh=0.5)
        self._apply_matches(matches, r_tracked_stracks, detections_second, activated_stracks, refind_stracks)

        # --- Occlusion handling for still-unmatched tracked tracks ---
        self._handle_occlusions(r_tracked_stracks, u_track, activated_stracks, lost_stracks)

        # --- Unconfirmed tracks (1-frame-old starts) ---
        detections = [detections[i] for i in u_detection]
        dists = self.get_dists(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # --- Init new tracks, suppressed by IoU against already-active tracks ---
        # Keep a plain list of box rows and only stack when we need to compute IoU.
        # Note: tracks confirmed for the first time this frame may have is_activated=False until
        # they hit min_track_len, so they won't appear here. New spawns within this loop are
        # appended below and will suppress each other regardless.
        active_boxes = [t.xyxy for t in activated_stracks if t.is_activated]
        suppress_on = self.init_iou_suppress < 1.0
        for inew in u_detection:
            det = detections[inew]
            if det.score < self.args.new_track_thresh:
                continue
            if suppress_on and active_boxes:
                stacked = np.asarray(active_boxes, dtype=np.float32)
                if matching.iou_matrix(det.xyxy[None, :], stacked).max() >= self.init_iou_suppress:
                    continue
            det.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(det)
            active_boxes.append(det.xyxy)

        # --- Retire lost tracks, with grace window for recently-occluded ones ---
        for track in self.lost_stracks:
            recently_occluded = track.was_recently_occluded and (
                self.frame_id - track.last_occluded_frame <= self.occ_reappear_window
            )
            if not recently_occluded and (self.frame_id - track.end_frame > self.max_frames_lost):
                track.mark_removed()
                removed_stracks.append(track)

        merge_track_pools(self, activated_stracks, refind_stracks, lost_stracks, removed_stracks)
        # Only emit tracks updated this frame: occluded tracks kept alive by `_handle_occlusions`
        # carry stale `idx` values from a previous frame's detection list, which would index into
        # today's results out-of-bounds.
        return np.asarray(
            [x.result for x in self.tracked_stracks if x.is_activated and x.frame_id == self.frame_id],
            dtype=np.float32,
        )

    def _handle_occlusions(self, r_tracked, u_track, activated_stracks, lost_stracks):
        """Flag unmatched tracked tracks as occluded when covered by an active neighbor.

        For each unmatched track, computes the fraction of its area covered by any currently-active
        track. If coverage exceeds ``occ_cover_thresh``, the track is marked occluded, its Kalman
        state is rolled back using the ring-buffer history (velocity from ``reset_velocity_offset_occ``
        frames ago, position from ``reset_pos_offset_occ`` frames ago), the bbox height is scaled by
        ``enlarge_bbox_occ`` once, and velocity is dampened by ``dampen_motion_occ``. Tracks that
        remain unmatched for more than two frames are transitioned to Lost, except while they are
        within the ``active_occ_to_lost_thresh`` occluded-grace window.

        Args:
            r_tracked (list[FastSTrack]): Candidate track pool.
            u_track (list[int] | np.ndarray): Indices into ``r_tracked`` of tracks still unmatched.
            activated_stracks (list[FastSTrack]): Tracks already matched this frame; used as the pool of potential
                occluders.
            lost_stracks (list[FastSTrack]): Output list; tracks transitioned to Lost are appended.
        """
        if len(u_track) == 0:
            return

        # Build active-track box array once (vectorized cover check).
        active = [t for t in activated_stracks if t.is_activated and not t.is_occluded]
        if active:
            active_boxes = np.asarray([t.xyxy for t in active], dtype=np.float32)
            active_ids = np.asarray([t.track_id for t in active])
        else:
            active_boxes = np.empty((0, 4), dtype=np.float32)
            active_ids = np.empty((0,), dtype=np.int64)

        unmatched = [r_tracked[i] for i in u_track]
        unmatched_boxes = (
            np.asarray([t.xyxy for t in unmatched], dtype=np.float32)
            if unmatched
            else np.empty((0, 4), dtype=np.float32)
        )

        if active_boxes.size and unmatched_boxes.size:
            cov = matching.coverage_matrix(unmatched_boxes, active_boxes)  # (U, A)
            # Avoid self-match: zero out columns that correspond to the same track id.
            unm_ids = np.asarray([t.track_id for t in unmatched])
            same = unm_ids[:, None] == active_ids[None, :]
            cov[same] = 0.0
            max_cov = cov.max(axis=1)
        else:
            max_cov = np.zeros(len(unmatched), dtype=np.float32)

        for i, track in enumerate(unmatched):
            track.not_matched += 1

            covered = bool(max_cov[i] > self.occ_cover_thresh)
            if covered and not track.is_occluded and track.state == TrackState.Tracked:
                track.is_occluded = True
                track.occluded_len = 1
                track.last_occluded_frame = self.frame_id
                track.was_recently_occluded = True

                hist = track.mean_history
                if track.mean is not None:
                    if len(hist) >= self.reset_velocity_offset_occ:
                        track.mean[4:8] = hist[-self.reset_velocity_offset_occ][4:8]
                    if len(hist) >= self.reset_pos_offset_occ:
                        track.mean[0:4] = hist[-self.reset_pos_offset_occ][0:4]
                    # Enlarge height once to expand search region.
                    track.mean[3] *= self.enlarge_bbox_occ
                    track.mean[4:8] *= self.dampen_motion_occ
            elif track.is_occluded:
                track.occluded_len += 1

            if track.was_recently_occluded and (self.frame_id - track.last_occluded_frame > self.occ_reappear_window):
                track.was_recently_occluded = False

            if track.state != TrackState.Lost:
                # Give occluded tracks a grace period before marking lost.
                if track.not_matched > 2 and (
                    not track.is_occluded or track.occluded_len > self.active_occ_to_lost_thresh
                ):
                    track.mark_lost()
                    lost_stracks.append(track)

    def _apply_matches(self, matches, pool, detections, activated_stracks, refind_stracks):
        """Apply matched ``(track, detection)`` pairs from an association pass.

        Updates tracked tracks in place, re-activates lost ones, and clears per-track occlusion
        bookkeeping on any successful match.

        Args:
            matches (Iterable[tuple[int, int]]): ``(track_idx, det_idx)`` pairs from :func:`matching.linear_assignment`.
            pool (list[FastSTrack]): Track pool indexed by ``track_idx``.
            detections (list[FastSTrack]): Detections indexed by ``det_idx``.
            activated_stracks (list[FastSTrack]): Output list for tracks updated from the Tracked state.
            refind_stracks (list[FastSTrack]): Output list for tracks re-activated from the Lost state.
        """
        for itracked, idet in matches:
            track = pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
            track.is_occluded = False
            track.not_matched = 0
            track.occluded_len = 0
