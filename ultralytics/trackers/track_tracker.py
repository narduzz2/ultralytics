# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""TrackTrack: Track-focused online multi-object tracker.

Reference: Shim et al., "Focusing on Tracks for Online Multi-Object Tracking" (CVPR 2025).

The tracker introduces two main components on top of the standard tracking-by-detection pipeline:
    - Track-Perspective-Based Association (TPA): multi-cue cost combining HMIoU, cosine ReID,
      confidence similarity, and corner angle distance, solved with an iterative assignment that
      progressively reduces the matching threshold.
    - Track-Aware Initialization (TAI): selectively creates new tracks by suppressing candidates
      that heavily overlap with existing active tracks or higher-scoring detections.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np
import scipy.linalg
import torch

from ..utils import LOGGER
from ..utils.ops import xywh2ltwh
from .basetrack import BaseTrack, TrackState
from .utils import matching
from .utils.gmc import GMC
from .utils.kalman_filter import KalmanFilterXYWH


def _nsa_kalman_update(kf, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray, confidence: float):
    """NSA-Kalman update: scale the innovation covariance by (1 - confidence).

    The standard Kalman update treats every measurement with the same noise model. NSA (Noise
    Scale Adaptive, from StrongSORT) scales the measurement noise down for high-confidence
    detections and up for low-confidence ones. In effect, a confident detection pulls the state
    estimate harder toward the observation, while an uncertain detection has less influence.

    Args:
        kf (KalmanFilterXYWH): Kalman filter providing the projection matrix and std weights.
        mean (np.ndarray): Predicted state mean.
        covariance (np.ndarray): Predicted state covariance.
        measurement (np.ndarray): Observed (x, y, w, h) box.
        confidence (float): Detection confidence in [0, 1]; higher = less measurement noise.

    Returns:
        new_mean (np.ndarray): Updated state mean.
        new_covariance (np.ndarray): Updated state covariance.
    """
    # Recompute projection so that NSA only scales the measurement noise R, not H P H^T
    std = [
        kf._std_weight_position * mean[2],
        kf._std_weight_position * mean[3],
        kf._std_weight_position * mean[2],
        kf._std_weight_position * mean[3],
    ]
    # Clamp confidence to avoid collapsing R to zero when the detector is over-confident
    scale = max(1.0 - float(confidence), 0.05)
    innovation_cov = np.diag(np.square(std)) * scale
    projected_mean = np.dot(kf._update_mat, mean)
    projected_cov = np.linalg.multi_dot((kf._update_mat, covariance, kf._update_mat.T)) + innovation_cov

    chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
    kalman_gain = scipy.linalg.cho_solve(
        (chol_factor, lower), np.dot(covariance, kf._update_mat.T).T, check_finite=False
    ).T
    innovation = measurement - projected_mean
    new_mean = mean + np.dot(innovation, kalman_gain.T)
    new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
    return new_mean, new_covariance


def _bbox_overlaps(a_xyxy: np.ndarray, b_xyxy: np.ndarray) -> np.ndarray:
    """Compute the IoU matrix between two sets of axis-aligned boxes.

    Args:
        a_xyxy (np.ndarray): Array of shape (N, 4) in (x1, y1, x2, y2) format.
        b_xyxy (np.ndarray): Array of shape (M, 4) in (x1, y1, x2, y2) format.

    Returns:
        (np.ndarray): IoU matrix of shape (N, M).
    """
    if len(a_xyxy) == 0 or len(b_xyxy) == 0:
        return np.zeros((len(a_xyxy), len(b_xyxy)), dtype=np.float64)

    inter_x1 = np.maximum(a_xyxy[:, 0:1], b_xyxy[:, 0:1].T)
    inter_y1 = np.maximum(a_xyxy[:, 1:2], b_xyxy[:, 1:2].T)
    inter_x2 = np.minimum(a_xyxy[:, 2:3], b_xyxy[:, 2:3].T)
    inter_y2 = np.minimum(a_xyxy[:, 3:4], b_xyxy[:, 3:4].T)

    inter = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
    area_a = (a_xyxy[:, 2] - a_xyxy[:, 0]) * (a_xyxy[:, 3] - a_xyxy[:, 1])
    area_b = (b_xyxy[:, 2] - b_xyxy[:, 0]) * (b_xyxy[:, 3] - b_xyxy[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / (union + 1e-9)


def _hmiou_distance(a_tracks: list, b_tracks: list) -> tuple[np.ndarray, np.ndarray]:
    """Compute HMIoU (Height-aware Modified IoU) similarity and distance.

    HMIoU = HIoU * IoU, where HIoU is the ratio of vertical overlap to total vertical extent. The
    height modifier improves matching when detections with similar aspect ratios are stacked
    vertically (common for pedestrians).

    Args:
        a_tracks (list): Objects with an `xyxy` property (e.g. tracks).
        b_tracks (list): Objects with an `xyxy` property (e.g. detections).

    Returns:
        iou_sim (np.ndarray): Raw IoU similarity matrix of shape (N, M).
        hmiou_dist (np.ndarray): HMIoU distance matrix of shape (N, M), equal to 1 - HMIoU.
    """
    if len(a_tracks) == 0 or len(b_tracks) == 0:
        n, m = len(a_tracks), len(b_tracks)
        return np.zeros((n, m), dtype=np.float64), np.ones((n, m), dtype=np.float64)

    a_boxes = np.ascontiguousarray([t.xyxy for t in a_tracks], dtype=np.float64)
    b_boxes = np.ascontiguousarray([t.xyxy for t in b_tracks], dtype=np.float64)

    iou_sim = _bbox_overlaps(a_boxes, b_boxes)

    # Height IoU: vertical overlap divided by vertical union
    h_overlap = np.minimum(a_boxes[:, 3:4], b_boxes[:, 3:4].T) - np.maximum(a_boxes[:, 1:2], b_boxes[:, 1:2].T)
    h_union = np.maximum(a_boxes[:, 3:4], b_boxes[:, 3:4].T) - np.minimum(a_boxes[:, 1:2], b_boxes[:, 1:2].T)
    h_iou = np.clip(h_overlap / (h_union + 1e-9), 0, 1)

    hmiou_sim = h_iou * iou_sim
    return iou_sim, 1.0 - hmiou_sim


def _corner_velocity(box_prev: np.ndarray, box_curr: np.ndarray) -> np.ndarray:
    """Compute normalized corner velocity vectors between two boxes.

    The box is decomposed into its four corners (LT, LB, RT, RB). Each corner's displacement
    between `box_prev` and `box_curr` is normalized to a unit direction vector.

    Args:
        box_prev (np.ndarray): Previous box in (x1, y1, x2, y2) format.
        box_curr (np.ndarray): Current box in (x1, y1, x2, y2) format.

    Returns:
        (np.ndarray): Corner velocity matrix of shape (4, 2) with unit-normalized (dx, dy) rows.
    """
    deltas = box_curr - box_prev
    corners = [
        (deltas[0], deltas[1]),  # LT
        (deltas[0], deltas[3]),  # LB
        (deltas[2], deltas[1]),  # RT
        (deltas[2], deltas[3]),  # RB
    ]
    vel = np.zeros((4, 2), dtype=np.float64)
    for i, (dx, dy) in enumerate(corners):
        norm = np.sqrt(dx**2 + dy**2) + 1e-5
        vel[i] = [dx / norm, dy / norm]
    return vel


def _angle_distance(tracks: list, dets: list, frame_id: int, delta_t: int = 3) -> np.ndarray:
    """Compute the angle distance between track velocities and track-to-detection directions.

    For each (track, detection) pair the per-corner direction is computed from the track's box at
    frame_id - delta_t to the detection, and compared via cosine angle to the track's stored corner
    velocity. The resulting distance is averaged over the 4 corners and weighted by detection score.

    Args:
        tracks (list[TTSTrack]): Tracks with `velocity` and box history.
        dets (list[TTSTrack]): Detection candidates.
        frame_id (int): Current frame ID.
        delta_t (int): Number of frames back used to look up the reference track box.

    Returns:
        (np.ndarray): Angle distance matrix of shape (N, M), each value in [0, 1].
    """
    n_tracks, n_dets = len(tracks), len(dets)
    if n_tracks == 0 or n_dets == 0:
        return np.ones((n_tracks, n_dets), dtype=np.float64)

    track_boxes_prev = np.stack([t.get_history_box(frame_id, delta_t) for t in tracks], axis=0)  # (N, 4)
    det_boxes = np.stack([d.xyxy for d in dets], axis=0)  # (M, 4)

    # Per-pair box delta: (N, M, 4)
    deltas = det_boxes[None, :, :] - track_boxes_prev[:, None, :]

    # Gather corner components so the last axis iterates LT, LB, RT, RB
    dx_idx = np.array([0, 0, 2, 2])
    dy_idx = np.array([1, 3, 1, 3])
    vel_dx = deltas[:, :, dx_idx]  # (N, M, 4)
    vel_dy = deltas[:, :, dy_idx]  # (N, M, 4)
    norms = np.sqrt(vel_dx * vel_dx + vel_dy * vel_dy) + 1e-5
    vel_dx /= norms
    vel_dy /= norms

    track_velocities = np.stack([t.velocity for t in tracks], axis=0)  # (N, 4, 2)

    # Per-corner dot product then average over corners
    dot = track_velocities[:, None, :, 0] * vel_dx + track_velocities[:, None, :, 1] * vel_dy
    angle_dist = np.abs(np.arccos(np.clip(dot, -1, 1))).mean(axis=-1) / np.pi  # (N, M)

    # Fuse with detection scores so low-confidence detections contribute less
    scores = np.array([d.score for d in dets])[None, :]
    angle_dist *= scores
    return angle_dist


def _confidence_distance(tracks: list, dets: list) -> np.ndarray:
    """Compute the confidence-based distance via linear projection of each track's score.

    Each track's projected score is `score + (score - prev_score)`, assuming a first-order trend.
    The distance is the absolute difference between the projected track score and each
    detection's confidence.

    Args:
        tracks (list[TTSTrack]): Tracks with `score` and `prev_score` attributes.
        dets (list[TTSTrack]): Detection candidates.

    Returns:
        (np.ndarray): Confidence distance matrix of shape (N, M).
    """
    if len(tracks) == 0 or len(dets) == 0:
        return np.ones((len(tracks), len(dets)), dtype=np.float64)

    t_score_prev = np.array([t.prev_score for t in tracks])
    t_score = np.array([t.score for t in tracks])
    t_score_proj = t_score + (t_score - t_score_prev)

    d_score = np.array([d.score for d in dets])
    return np.abs(t_score_proj[:, None] - d_score[None, :])


def _iterative_associate(cost: np.ndarray, match_thr: float, reduce_step: float = 0.05):
    """Greedily match tracks to detections with a progressively relaxed threshold.

    At each iteration the minimum-cost mutually-nearest pairs below `match_thr` are added to the
    matches, those rows/columns are masked out, and `match_thr` is reduced by `reduce_step`. This
    prioritizes high-confidence matches while still allowing weaker associations to be made in
    later iterations.

    Args:
        cost (np.ndarray): Cost matrix of shape (N, M) with values in [0, 1].
        match_thr (float): Initial matching threshold.
        reduce_step (float): Amount to reduce the threshold after each iteration.

    Returns:
        matches (list[list[int]]): List of matched (track_idx, det_idx) pairs.
        u_tracks (list[int]): Indices of unmatched tracks.
        u_dets (list[int]): Indices of unmatched detections.
    """
    matches = []
    cost = cost.copy()

    while True:
        new_matches = []
        if cost.shape[0] > 0 and cost.shape[1] > 0:
            min_det_per_track = np.argmin(cost, axis=1)
            min_track_per_det = np.argmin(cost, axis=0)
            for t_idx, d_idx in enumerate(min_det_per_track):
                if min_track_per_det[d_idx] == t_idx and cost[t_idx, d_idx] < match_thr:
                    new_matches.append([t_idx, d_idx])

        if len(new_matches) == 0:
            break

        matches.extend(new_matches)
        for t, d in new_matches:
            cost[t, :] = 1.0
            cost[:, d] = 1.0
        match_thr -= reduce_step

    m_tracks = {t for t, _ in matches}
    m_dets = {d for _, d in matches}
    u_tracks = [t for t in range(cost.shape[0]) if t not in m_tracks]
    u_dets = [d for d in range(cost.shape[1]) if d not in m_dets]
    return matches, u_tracks, u_dets


def attach_raw_preds_hook(predictor) -> None:
    """Wrap `predictor.postprocess` so raw (pre-NMS) predictions are captured for D_del.

    TrackTrack needs access to the pre-NMS model output to run the looser secondary NMS pass
    (paper Eq. 1). The generic tracking callback is kept TrackTrack-agnostic; the hook lives here.
    Idempotent — if already wrapped, returns without re-wrapping.

    Args:
        predictor (BasePredictor): Predictor to instrument; `_raw_preds` and `_preproc_img_shape`
            are populated on each postprocess call after wrapping.
    """
    if hasattr(predictor, "_orig_postprocess"):
        return
    orig = predictor.postprocess

    @wraps(orig)
    def _wrapped(preds, img, *args, **kwargs):
        predictor._raw_preds = preds.clone() if isinstance(preds, torch.Tensor) else preds
        predictor._preproc_img_shape = img.shape[2:]  # (H, W) of preprocessed image
        return orig(preds, img, *args, **kwargs)

    predictor._orig_postprocess = orig
    predictor.postprocess = _wrapped


def compute_dets_del(predictor) -> list | None:
    """Compute TrackTrack's D_del set from the raw predictions captured by the postprocess hook.

    D_del are high-confidence detections that the tight NMS suppressed but which survive a looser
    NMS pass at IoU=0.95. TrackTrack then treats these as additional low-priority candidates
    during association (paper Eq. 1).

    Args:
        predictor (BasePredictor): Predictor previously instrumented with `attach_raw_preds_hook`.

    Returns:
        (list[tuple | None] | None): Per-batch-element `(xywh, conf, cls)` bundles (or None where
            no deleted detections were found). Returns None if raw preds were not captured.
    """
    raw_preds = getattr(predictor, "_raw_preds", None)
    if raw_preds is None:
        return None

    from torchvision.ops import box_iou

    from ultralytics.utils import nms, ops

    is_obb = predictor.args.task == "obb"
    preds_loose = nms.non_max_suppression(
        raw_preds,
        predictor.args.conf,
        0.95,  # loose IoU; paper pairs tight 0.80 + loose 0.95
        predictor.args.classes,
        predictor.args.agnostic_nms,
        max_det=predictor.args.max_det,
        nc=0 if predictor.args.task == "detect" else len(predictor.model.names),
        end2end=getattr(predictor.model, "end2end", False),
        rotated=is_obb,
    )

    dets_del_list = []
    im_shape = getattr(predictor, "_preproc_img_shape", None)
    for loose, result in zip(preds_loose, predictor.results):
        det_boxes = (result.obb if is_obb else result.boxes).cpu()
        if len(loose) == 0 or len(det_boxes) == 0:
            dets_del_list.append(None)
            continue

        loose_scaled = loose.clone()
        if im_shape is not None:
            loose_scaled[:, :4] = ops.scale_boxes(im_shape, loose_scaled[:, :4], result.orig_shape)

        tight_xyxy = det_boxes.xyxy.cpu()
        loose_xyxy = loose_scaled[:, :4].cpu()
        if tight_xyxy.numel() == 0 or loose_xyxy.numel() == 0:
            dets_del_list.append(None)
            continue

        ious = box_iou(loose_xyxy, tight_xyxy)
        max_iou, _ = ious.max(dim=1)
        del_mask = max_iou < 0.97
        if not del_mask.any():
            dets_del_list.append(None)
            continue

        del_boxes = loose_scaled[del_mask]
        del_xywh = ops.xyxy2xywh(del_boxes[:, :4])
        dets_del_list.append((del_xywh.numpy(), del_boxes[:, 4].numpy(), del_boxes[:, 5].numpy()))

    predictor._raw_preds = None  # consumed
    return dets_del_list


def _track_aware_nms(tracks: list, dets: list, tai_thr: float, init_thr: float) -> list[bool]:
    """Apply Track-Aware Initialization NMS to filter candidates for new-track creation.

    A detection is allowed to spawn a new track only if its score is above `init_thr`, it does not
    overlap an existing active track by more than `tai_thr`, and it is not overlapped by any
    higher-scoring detection that is also about to start a track.

    Args:
        tracks (list): Currently active tracks.
        dets (list): Detection candidates for new track initialization.
        tai_thr (float): IoU threshold used to suppress overlapping candidates.
        init_thr (float): Minimum detection score required to consider spawning a new track.

    Returns:
        (list[bool]): One flag per detection indicating whether it may initialize a new track.
    """
    if len(dets) == 0:
        return []

    scores = np.array([d.score for d in dets])
    allow = list(scores > init_thr)

    all_objs = tracks + dets
    if len(all_objs) < 2:
        return allow

    all_boxes = np.ascontiguousarray([o.xyxy for o in all_objs], dtype=np.float64)
    pair_iou = _bbox_overlaps(all_boxes, all_boxes)
    n_tracks = len(tracks)

    for i in range(len(dets)):
        if not allow[i]:
            continue
        # Suppress if heavily overlapping an existing track
        if n_tracks > 0 and np.max(pair_iou[n_tracks + i, :n_tracks]) > tai_thr:
            allow[i] = False
            continue
        # Suppress lower-scoring neighbors that heavily overlap this detection
        for j in range(len(dets)):
            if i != j and allow[j] and scores[i] > scores[j]:
                if pair_iou[n_tracks + i, n_tracks + j] > tai_thr:
                    allow[j] = False

    return allow


class TTSTrack(BaseTrack):
    """Single-object track for TrackTrack with corner velocity, score history, and ReID features.

    Extends BaseTrack with state needed for TrackTrack's multi-cue association: per-corner unit
    velocity vectors used by the angle distance, previous-frame score for confidence projection,
    a per-frame box history used for velocity lookback, and optional EMA-smoothed ReID features.

    Attributes:
        shared_kalman (KalmanFilterXYWH): Shared Kalman filter instance used for batch prediction.
        _tlwh (np.ndarray): Initial box in top-left-width-height format.
        kalman_filter (KalmanFilterXYWH): Per-track Kalman filter used after activation.
        mean (np.ndarray): Mean state vector of the Kalman filter.
        covariance (np.ndarray): Covariance matrix of the Kalman filter.
        is_activated (bool): Whether the track has been promoted from new to active.
        score (float): Current detection confidence.
        prev_score (float): Confidence on the previous update (for score projection).
        tracklet_len (int): Number of successful updates since activation.
        cls (Any): Class label.
        idx (int): Index of the originating detection within the current frame.
        frame_id (int): Frame ID of the most recent update.
        start_frame (int): Frame in which this track was first created.
        angle (float | None): Rotation angle for oriented boxes, else None.
        velocity (np.ndarray): Per-corner unit velocity vectors of shape (4, 2).
        delta_t (int): Look-back window used when constructing corner velocities.
        smooth_feat (np.ndarray | None): EMA-smoothed ReID embedding.
        curr_feat (np.ndarray | None): Raw ReID embedding from the current frame.

    Methods:
        update_features: Update `smooth_feat` and `curr_feat` with EMA smoothing.
        get_history_box: Retrieve a historical box, falling back to the most recent available frame.
        predict: Run single-track Kalman prediction.
        multi_predict: Run batched Kalman prediction for a list of tracks.
        multi_gmc: Apply a global motion warp to all track states.
        activate: Initialize a brand new track.
        re_activate: Reactivate a lost track with a fresh detection.
        update: Update a matched track with a new detection.

    Examples:
        Create and activate a new track
        >>> track = TTSTrack(xywh=[100, 200, 50, 80, 0], score=0.9, cls="person")
        >>> track.activate(kalman_filter=KalmanFilterXYWH(), frame_id=1)
    """

    shared_kalman = KalmanFilterXYWH()
    min_track_len = 3  # class-level default; overridden by TRACKTRACK.__init__ from config

    def __init__(self, xywh: list[float], score: float, cls: Any, feat: np.ndarray | None = None):
        """Initialize a new TTSTrack instance.

        Args:
            xywh (list[float]): Bounding box in `(x, y, w, h, idx)` or `(x, y, w, h, angle, idx)`
                format, where (x, y) is the center, (w, h) are width and height, and `idx` is the
                detection index in the current frame.
            score (float): Detection confidence score.
            cls (Any): Class label for the detected object.
            feat (np.ndarray | None): Optional ReID feature vector attached to this detection.
        """
        super().__init__()
        assert len(xywh) in {5, 6}, f"expected 5 or 6 values but got {len(xywh)}"
        self._tlwh = np.asarray(xywh2ltwh(xywh[:4]), dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.prev_score = score
        self.tracklet_len = 0
        self.cls = cls
        self.idx = xywh[-1]
        self.angle = xywh[4] if len(xywh) == 6 else None

        # Per-corner unit velocity vectors (LT, LB, RT, RB) and lookback window
        self.velocity = np.zeros((4, 2), dtype=np.float64)
        self.delta_t = 3

        # Frame-indexed history: frame_id -> (box_xyxy, score, mean, covariance)
        self._history: dict[int, tuple] = {}

        # ReID feature state
        self.smooth_feat = None
        self.curr_feat = None
        self._alpha = 0.95
        if feat is not None:
            self.update_features(feat)

    def update_features(self, feat: np.ndarray):
        """Update the ReID feature cache with exponential moving average smoothing.

        The smoothing factor beta adapts with the current detection score so high-confidence
        observations have slightly more influence on the long-term feature.

        Args:
            feat (np.ndarray): New (unnormalized) feature vector.
        """
        feat = feat / (np.linalg.norm(feat) + 1e-9)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat.copy()
        else:
            beta = self._alpha + (1 - self._alpha) * (1 - self.score)
            self.smooth_feat = beta * self.smooth_feat + (1 - beta) * feat
            self.smooth_feat /= np.linalg.norm(self.smooth_feat) + 1e-9

    def get_history_box(self, frame_id: int, dt: int) -> np.ndarray:
        """Return the historical box from `dt` frames before `frame_id`.

        Falls back to the most recent stored box if no exact match exists, and to the current box
        if the history is empty (new tracks).

        Args:
            frame_id (int): Current frame ID.
            dt (int): Number of frames to look back.

        Returns:
            (np.ndarray): Historical box in (x1, y1, x2, y2) format.
        """
        target = frame_id - dt
        if target in self._history:
            return self._history[target][0].copy()
        if self._history:
            return self._history[max(self._history.keys())][0].copy()
        return self.xyxy.copy()

    def predict(self):
        """Predict the next state (mean and covariance) of the track using the Kalman filter."""
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks: list[TTSTrack]):
        """Run batched Kalman prediction over the provided list of tracks."""
        if len(stracks) <= 0:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][6] = 0
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = TTSTrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks: list[TTSTrack], H: np.ndarray = np.eye(2, 3)):
        """Apply a global motion compensation warp to every track's mean and covariance.

        Args:
            stracks (list[TTSTrack]): Tracks to warp in place.
            H (np.ndarray): 2x3 affine transform describing camera motion between frames.
        """
        if stracks:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter: KalmanFilterXYWH, frame_id: int):
        """Activate a brand new tracklet and initialize its Kalman state and history entry.

        Args:
            kalman_filter (KalmanFilterXYWH): Kalman filter instance to attach to this track.
            frame_id (int): Current frame ID.
        """
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.convert_coords(self._tlwh))
        self._history[frame_id] = (self.xyxy.copy(), self.score, self.mean.copy(), self.covariance.copy())

        self.tracklet_len = 0
        self.state = TrackState.New
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track: TTSTrack, frame_id: int, new_id: bool = False):
        """Reactivate a previously lost track by absorbing a newly matched detection.

        Args:
            new_track (TTSTrack): Detection object that matched this lost track.
            frame_id (int): Current frame ID.
            new_id (bool): If True, assign a fresh track ID rather than reusing the old one.
        """
        self.prev_score = self.score
        # NSA-Kalman update weights the measurement by detection confidence
        self.mean, self.covariance = _nsa_kalman_update(
            self.kalman_filter, self.mean, self.covariance, self.convert_coords(new_track.tlwh), new_track.score
        )
        self._history[frame_id] = (self.xyxy.copy(), new_track.score, self.mean.copy(), self.covariance.copy())
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.cls = new_track.cls
        self.angle = new_track.angle
        self.idx = new_track.idx

    def update(self, new_track: TTSTrack, frame_id: int):
        """Update a matched track with a fresh detection, refreshing velocity and ReID features.

        Promotes the track from `New` to `Tracked` only after `min_track_len` successful matches,
        matching the paper's requirement for a stable track before it enters the output.

        Args:
            new_track (TTSTrack): Matched detection for this frame.
            frame_id (int): Current frame ID.
        """
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.prev_score = self.score

        new_tlwh = new_track.tlwh
        # NSA-Kalman update weights the measurement by detection confidence (StrongSORT)
        self.mean, self.covariance = _nsa_kalman_update(
            self.kalman_filter, self.mean, self.covariance, self.convert_coords(new_tlwh), new_track.score
        )
        self._history[frame_id] = (new_track.xyxy.copy(), new_track.score, self.mean.copy(), self.covariance.copy())

        # Recompute corner velocity as an average of 1..delta_t-frame lookbacks
        self.velocity = np.zeros((4, 2), dtype=np.float64)
        for dt in range(1, self.delta_t + 1):
            prev_box = self.get_history_box(frame_id, dt)
            self.velocity += _corner_velocity(prev_box, new_track.xyxy) / dt
        self.velocity /= self.delta_t

        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

        # Promote to Tracked only after min_track_len consecutive matches
        if self.state == TrackState.Tracked or self.tracklet_len >= self.min_track_len:
            self.state = TrackState.Tracked
            self.is_activated = True
        self.score = new_track.score
        self.cls = new_track.cls
        self.angle = new_track.angle
        self.idx = new_track.idx

    def convert_coords(self, tlwh: np.ndarray) -> np.ndarray:
        """Convert a top-left-width-height box to the center-x-center-y-width-height format used by the filter."""
        return self.tlwh_to_xywh(tlwh)

    @property
    def tlwh(self) -> np.ndarray:
        """Get the current bounding box in (top-left x, top-left y, width, height) format."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def xyxy(self) -> np.ndarray:
        """Get the current bounding box in (min x, min y, max x, max y) format."""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh: np.ndarray) -> np.ndarray:
        """Convert a tlwh box to (center x, center y, width, height) format."""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    @property
    def xywh(self) -> np.ndarray:
        """Get the current bounding box in (center x, center y, width, height) format."""
        ret = np.asarray(self.tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    @property
    def xywha(self) -> np.ndarray:
        """Get position in (center x, center y, width, height, angle) format; falls back to xywh if angle is missing."""
        if self.angle is None:
            LOGGER.warning("`angle` attr not found, returning `xywh` instead.")
            return self.xywh
        return np.concatenate([self.xywh, self.angle[None]])

    @property
    def result(self) -> list[float]:
        """Get the packed tracking result `[*coords, track_id, score, cls, idx]` for downstream use."""
        coords = self.xyxy if self.angle is None else self.xywha
        return [*coords.tolist(), self.track_id, self.score, self.cls, self.idx]

    def __repr__(self) -> str:
        """Return a short string representation showing the track ID and its active frame range."""
        return f"TT_{self.track_id}_({self.start_frame}-{self.end_frame})"


class TRACKTRACK:
    """TrackTrack: multi-object tracker based on Track-Perspective Association and Track-Aware Init.

    This class implements the full algorithm from Shim et al. (CVPR 2025). Detections are split
    into high-confidence, low-confidence, and deleted (loose-NMS recovered) sets. A multi-cue cost
    matrix is built from HMIoU, cosine ReID similarity, confidence projection, and corner angle
    distance, then solved with iterative assignment. Unmatched high-confidence detections are
    passed through Track-Aware Initialization NMS before creating new tracks.

    Attributes:
        tracked_stracks (list[TTSTrack]): Currently active tracks.
        lost_stracks (list[TTSTrack]): Tracks that are temporarily lost (still within track_buffer).
        removed_stracks (list[TTSTrack]): Tracks that have been removed.
        frame_id (int): Current frame ID.
        args (Namespace): Parsed tracker configuration.
        max_time_lost (int): Maximum number of frames a lost track is kept before removal.
        kalman_filter (KalmanFilterXYWH): Kalman filter used when activating new tracks.
        det_thr (float): Score threshold that separates D_high from D_low.
        match_thr (float): Starting threshold for iterative assignment.
        penalty_p (float): Cost penalty added to D_low columns (paper τ_p).
        penalty_q (float): Cost penalty added to D_del columns (paper τ_q).
        reduce_step (float): Amount to reduce match_thr between iterative-assignment rounds.
        iou_weight (float): Weight for the HMIoU distance term in the cost matrix.
        reid_weight (float): Weight for the cosine ReID distance term (or HMIoU fallback).
        conf_weight (float): Weight for the confidence projection distance.
        angle_weight (float): Weight for the corner angle distance.
        tai_thr (float): IoU threshold used inside Track-Aware Initialization NMS.
        init_thr (float): Minimum score for a detection to be allowed to start a new track.
        gmc (GMC): Global motion compensation module used to warp tracks between frames.
        with_reid (bool): Whether ReID features are used to refine the cost matrix.
        encoder (Callable | None): ReID feature extractor when `with_reid` is True.

    Methods:
        update: Advance the tracker one frame and return the active tracked objects.
        get_kalmanfilter: Return a Kalman filter instance used for new tracks.
        init_track: Build TTSTrack detection objects from raw results.
        multi_predict: Run batched Kalman prediction across a list of tracks.
        reset_id: Reset the global track ID counter.
        reset: Clear all tracker state.
        joint_stracks: Merge two track lists while keeping track IDs unique.
        sub_stracks: Remove tracks present in one list from another.
        remove_duplicate_stracks: Resolve overlapping tracks by keeping the older one.

    Examples:
        Initialize TrackTrack and process a frame of detections
        >>> tracker = TRACKTRACK(args, frame_rate=30)
        >>> results = yolo_model.detect(image)
        >>> tracked_objects = tracker.update(results, img=image)
    """

    def __init__(self, args, frame_rate: int = 30):
        """Initialize a TRACKTRACK instance.

        Args:
            args (Namespace): Parsed tracker configuration containing thresholds, cost weights,
                GMC parameters, and ReID options. See `ultralytics/cfg/trackers/tracktrack.yaml`.
            frame_rate (int): Frame rate of the input video, used to scale the lost-track buffer.
        """
        self.tracked_stracks: list[TTSTrack] = []
        self.lost_stracks: list[TTSTrack] = []
        self.removed_stracks: list[TTSTrack] = []

        self.frame_id = 0
        self.args = args
        self.max_time_lost = int(frame_rate / 30.0 * args.track_buffer)
        self.kalman_filter = self.get_kalmanfilter()
        self.reset_id()

        # Cost matrix weights and iterative assignment parameters
        self.det_thr = getattr(args, "det_thr", 0.6)
        self.match_thr = getattr(args, "match_thresh", 0.7)
        self.penalty_p = getattr(args, "penalty_p", 0.2)
        self.penalty_q = getattr(args, "penalty_q", 0.4)
        self.reduce_step = getattr(args, "reduce_step", 0.05)
        self.min_track_len = getattr(args, "min_track_len", 3)
        TTSTrack.min_track_len = self.min_track_len  # propagate to track instances
        self.iou_weight = getattr(args, "iou_weight", 0.5)
        self.reid_weight = getattr(args, "reid_weight", 0.5)
        self.conf_weight = getattr(args, "conf_weight", 0.1)
        self.angle_weight = getattr(args, "angle_weight", 0.05)

        # Track-Aware Initialization parameters
        self.tai_thr = getattr(args, "tai_thr", 0.55)
        self.init_thr = getattr(args, "init_thr", 0.7)

        # Global motion compensation; maxCorners/downscale/skip are tunable for speed
        gmc_method = getattr(args, "gmc_method", "sparseOptFlow")
        gmc_downscale = getattr(args, "gmc_downscale", 3)
        self.gmc = GMC(method=gmc_method, downscale=gmc_downscale)
        if gmc_method == "sparseOptFlow":
            gmc_max_corners = getattr(args, "gmc_max_corners", 200)
            self.gmc.feature_params["maxCorners"] = gmc_max_corners
        self._gmc_skip = getattr(args, "gmc_skip_frames", 0)
        self._gmc_warp_cached = np.eye(2, 3, dtype=np.float64)
        self._gmc_counter = 0

        # ReID encoder: "auto" uses the detector's backbone features; a path loads an external model
        self.with_reid = getattr(args, "with_reid", False)
        self.encoder = None
        if self.with_reid:
            model = getattr(args, "model", "auto")
            if model == "auto":
                self.encoder = lambda feats, s: [f.cpu().numpy() for f in feats]
            else:
                from .bot_sort import ReID

                self.encoder = ReID(model)

    def update(
        self, results, img: np.ndarray | None = None, feats: np.ndarray | None = None, dets_del=None
    ) -> np.ndarray:
        """Advance the tracker by one frame and return the active tracked objects.

        Args:
            results (object): Detection results exposing `conf`, `cls`, and either `xywh` or
                `xywhr` (oriented boxes). Indexable by class labels returned from YOLO.
            img (np.ndarray | None): Current frame as a (H, W, 3) BGR image, required for GMC and
                ReID feature extraction.
            feats (np.ndarray | None): Optional backbone features when the ReID encoder is in
                "auto" mode.
            dets_del (tuple | None): Optional deleted-detection bundle `(xywh, conf, cls)` produced
                by a looser secondary NMS pass (paper's D_del).

        Returns:
            (np.ndarray): Array of shape (N, 8) with rows `[x1, y1, x2, y2, id, score, cls, idx]`.
        """
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        scores = results.conf
        bboxes = results.xywhr if hasattr(results, "xywhr") else results.xywh
        bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)

        # Partition detections into high and low confidence subsets
        remain_inds = scores >= self.args.track_high_thresh
        inds_low = (scores > self.args.track_low_thresh) & (scores < self.args.track_high_thresh)

        bboxes_keep = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        cls_keep = results.cls[remain_inds]

        bboxes_second = bboxes[inds_low]
        scores_second = scores[inds_low]
        cls_second = results.cls[inds_low]

        # Build detection objects; attach ReID features to high-confidence detections when enabled
        if self.with_reid and self.encoder is not None and img is not None and len(bboxes_keep) > 0:
            features = self.encoder(img, bboxes_keep)  # external ReID encoder crops from the full image
            dets_high = [TTSTrack(b, s, c, f) for b, s, c, f in zip(bboxes_keep, scores_keep, cls_keep, features)]
        else:
            dets_high = [TTSTrack(b, s, c) for b, s, c in zip(bboxes_keep, scores_keep, cls_keep)]
        dets_low = [TTSTrack(b, s, c) for b, s, c in zip(bboxes_second, scores_second, cls_second)]

        # D_del: high-confidence detections recovered from a looser NMS pass (paper Eq. 1)
        dets_del_high = []
        if dets_del is not None:
            del_xywh, del_conf, del_cls = dets_del
            del_bboxes = np.concatenate([del_xywh, -np.ones((len(del_xywh), 1))], axis=-1)
            del_high_mask = del_conf > self.det_thr
            if del_high_mask.any():
                dets_del_high = [
                    TTSTrack(b, s, c)
                    for b, s, c in zip(del_bboxes[del_high_mask], del_conf[del_high_mask], del_cls[del_high_mask])
                ]

        # Separate activated (tracked/lost) tracks from new unconfirmed ones
        tracked_stracks = []
        unconfirmed = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)

        # Apply global motion compensation; optionally reuse the previous warp for several frames
        if img is not None:
            if self._gmc_skip > 0 and self._gmc_counter % (self._gmc_skip + 1) != 0:
                warp = self._gmc_warp_cached
            else:
                warp = self.gmc.apply(img, [t.xyxy for t in dets_high])
                self._gmc_warp_cached = warp
            self._gmc_counter += 1
            TTSTrack.multi_gmc(strack_pool, warp)
            TTSTrack.multi_gmc(unconfirmed, warp)

        # Kalman prediction before association
        self.multi_predict(strack_pool)

        # Main association: tracked/lost vs the combined D_high + D_low + D_del (paper Eq. 1)
        all_dets = dets_high + dets_low + dets_del_high
        n_high = len(dets_high)
        n_low = len(dets_low)

        iou_sim, hmiou_dist = _hmiou_distance(strack_pool, all_dets)
        cost = self.iou_weight * hmiou_dist
        if self.with_reid and self.encoder is not None:
            cost += self.reid_weight * self._cosine_distance(strack_pool, all_dets)
        else:
            cost += self.reid_weight * hmiou_dist  # fall back to HMIoU when no ReID is available
        cost += self.conf_weight * _confidence_distance(strack_pool, all_dets)
        cost += self.angle_weight * _angle_distance(strack_pool, all_dets, self.frame_id)

        # Column-wise penalties for low-confidence and deleted detections
        if cost.shape[1] > n_high:
            cost[:, n_high : n_high + n_low] += self.penalty_p  # τ_p for D_low
        if dets_del_high and cost.shape[1] > n_high + n_low:
            cost[:, n_high + n_low :] += self.penalty_q  # τ_q for D_del

        # Force dissociation where the raw IoU is negligible
        if iou_sim.size > 0:
            cost[iou_sim <= 0.10] = 1.0
        cost = np.clip(cost, 0, 1)

        matches, u_track, u_det = _iterative_associate(cost, self.match_thr, self.reduce_step)

        for t_idx, d_idx in matches:
            track = strack_pool[t_idx]
            det = all_dets[d_idx]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for t_idx in u_track:
            track = strack_pool[t_idx]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # Second association: unconfirmed tracks vs leftover high-confidence detections, using the
        # same multi-cue cost as the main stage so ReID information is not discarded.
        remaining_high_dets = [all_dets[i] for i in u_det if i < n_high]
        if unconfirmed and remaining_high_dets:
            uc_iou_sim, uc_hmiou_dist = _hmiou_distance(unconfirmed, remaining_high_dets)
            uc_cost = self.iou_weight * uc_hmiou_dist
            if self.with_reid and self.encoder is not None:
                uc_cost += self.reid_weight * self._cosine_distance(unconfirmed, remaining_high_dets)
            else:
                uc_cost += self.reid_weight * uc_hmiou_dist
            uc_cost += self.conf_weight * _confidence_distance(unconfirmed, remaining_high_dets)
            uc_cost += self.angle_weight * _angle_distance(unconfirmed, remaining_high_dets, self.frame_id)
            if uc_iou_sim.size > 0:
                uc_cost[uc_iou_sim <= 0.10] = 1.0
            uc_cost = np.clip(uc_cost, 0, 1)
            uc_matches, uc_u_track, uc_u_det = _iterative_associate(uc_cost, self.match_thr, self.reduce_step)
            for t_idx, d_idx in uc_matches:
                unconfirmed[t_idx].update(remaining_high_dets[d_idx], self.frame_id)
                activated_stracks.append(unconfirmed[t_idx])
            for t_idx in uc_u_track:
                unconfirmed[t_idx].mark_removed()
                removed_stracks.append(unconfirmed[t_idx])
            remaining_high_dets = [remaining_high_dets[i] for i in uc_u_det]
        else:
            for track in unconfirmed:
                track.mark_removed()
                removed_stracks.append(track)

        # Track-Aware Initialization: spawn new tracks only from survivors of TAI NMS
        active_tracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        active_tracks.extend(activated_stracks)
        allow_flags = _track_aware_nms(active_tracks, remaining_high_dets, self.tai_thr, self.init_thr)
        for i, det in enumerate(remaining_high_dets):
            if i < len(allow_flags) and allow_flags[i]:
                det.activate(self.kalman_filter, self.frame_id)
                activated_stracks.append(det)

        # Remove lost tracks that have exceeded the buffer
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # Bookkeeping: merge/filter track lists and bound the removed-tracks history
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)
        self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        self.removed_stracks.extend(removed_stracks)
        if len(self.removed_stracks) > 1000:
            self.removed_stracks = self.removed_stracks[-1000:]

        return np.asarray(
            [x.result for x in self.tracked_stracks if x.is_activated and x.frame_id == self.frame_id],
            dtype=np.float32,
        )

    @staticmethod
    def _cosine_distance(tracks: list[TTSTrack], dets: list[TTSTrack]) -> np.ndarray:
        """Compute pairwise cosine distance between track and detection ReID embeddings.

        Missing embeddings are substituted with zero vectors of the detected feature dimension so
        rows/columns without features contribute a constant maximum distance.

        Args:
            tracks (list[TTSTrack]): Tracks carrying an EMA-smoothed `smooth_feat`.
            dets (list[TTSTrack]): Detections carrying `curr_feat`.

        Returns:
            (np.ndarray): Cosine distance matrix of shape (N, M) clipped to [0, 1].
        """
        if len(tracks) == 0 or len(dets) == 0:
            return np.ones((len(tracks), len(dets)), dtype=np.float64)

        dim = 128  # fallback feature dim if no embeddings have been populated yet
        for obj in (*tracks, *dets):
            f = obj.smooth_feat if obj.smooth_feat is not None else obj.curr_feat
            if f is not None:
                dim = f.shape[0]
                break

        t_feat = np.stack(
            [t.smooth_feat if t.smooth_feat is not None else np.zeros(dim, dtype=np.float32) for t in tracks]
        )
        d_feat = np.stack(
            [d.curr_feat if d.curr_feat is not None else np.zeros(dim, dtype=np.float32) for d in dets]
        )
        return np.clip(1 - np.dot(t_feat, d_feat.T), 0, 1)

    def get_kalmanfilter(self) -> KalmanFilterXYWH:
        """Return the Kalman filter instance used when activating new tracks."""
        return KalmanFilterXYWH()

    def init_track(self, results, img: np.ndarray | None = None) -> list[TTSTrack]:
        """Build TTSTrack detection objects from raw YOLO results."""
        if len(results) == 0:
            return []
        bboxes = results.xywhr if hasattr(results, "xywhr") else results.xywh
        bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)
        return [TTSTrack(xywh, s, c) for (xywh, s, c) in zip(bboxes, results.conf, results.cls)]

    def multi_predict(self, tracks: list[TTSTrack]):
        """Run batched Kalman prediction for the provided list of tracks."""
        TTSTrack.multi_predict(tracks)

    @staticmethod
    def reset_id():
        """Reset the global TTSTrack ID counter."""
        TTSTrack.reset_id()

    def reset(self):
        """Reset all tracker state, including GMC warp history and the ID counter."""
        self.tracked_stracks: list[TTSTrack] = []
        self.lost_stracks: list[TTSTrack] = []
        self.removed_stracks: list[TTSTrack] = []
        self.frame_id = 0
        self.kalman_filter = self.get_kalmanfilter()
        self.reset_id()
        self.gmc.reset_params()

    @staticmethod
    def joint_stracks(tlista: list[TTSTrack], tlistb: list[TTSTrack]) -> list[TTSTrack]:
        """Concatenate two track lists while ensuring track IDs remain unique."""
        exists = {}
        res = []
        for t in tlista:
            exists[t.track_id] = 1
            res.append(t)
        for t in tlistb:
            if not exists.get(t.track_id, 0):
                exists[t.track_id] = 1
                res.append(t)
        return res

    @staticmethod
    def sub_stracks(tlista: list[TTSTrack], tlistb: list[TTSTrack]) -> list[TTSTrack]:
        """Return tlista with every track present in tlistb removed."""
        track_ids_b = {t.track_id for t in tlistb}
        return [t for t in tlista if t.track_id not in track_ids_b]

    @staticmethod
    def remove_duplicate_stracks(
        stracksa: list[TTSTrack], stracksb: list[TTSTrack]
    ) -> tuple[list[TTSTrack], list[TTSTrack]]:
        """Remove duplicates by keeping the longer-lived track for each overlapping pair."""
        pdist = matching.iou_distance(stracksa, stracksb)
        pairs = np.where(pdist < 0.15)
        dupa, dupb = [], []
        for p, q in zip(*pairs):
            timep = stracksa[p].frame_id - stracksa[p].start_frame
            timeq = stracksb[q].frame_id - stracksb[q].start_frame
            if timep > timeq:
                dupb.append(q)
            else:
                dupa.append(p)
        resa = [t for i, t in enumerate(stracksa) if i not in dupa]
        resb = [t for i, t in enumerate(stracksb) if i not in dupb]
        return resa, resb
