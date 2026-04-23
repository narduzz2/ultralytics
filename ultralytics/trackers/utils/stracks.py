# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Shared helpers for operating on lists of track objects across trackers.

These functions are intentionally generic: they only touch attributes that every track
implementation exposes (`track_id`, `frame_id`, `start_frame`, `xyxy`, `mean`, `covariance`).
"""

from __future__ import annotations

import numpy as np

from . import matching


def joint_stracks(atracks: list, btracks: list) -> list:
    """Combine two track lists into one, de-duplicating by `track_id`.

    Args:
        atracks (list[STrack]): First list of tracks; entries win on `track_id` collisions.
        btracks (list[STrack]): Second list of tracks.

    Returns:
        (list[STrack]): Union of `atracks` and `btracks` with duplicate `track_id`s removed.

    Examples:
        Merge the currently tracked pool with newly activated tracks
        >>> merged = joint_stracks(tracked_stracks, activated_stracks)
    """
    exists = {}
    res = []
    for t in atracks:
        exists[t.track_id] = 1
        res.append(t)
    for t in btracks:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(atracks: list, btracks: list) -> list:
    """Filter out tracks from `atracks` whose `track_id` appears in `btracks`.

    Args:
        atracks (list[STrack]): Source list of tracks to filter.
        btracks (list[STrack]): Tracks whose `track_id`s should be excluded from the output.

    Returns:
        (list[STrack]): Elements of `atracks` whose `track_id` is not present in `btracks`.

    Examples:
        Remove any re-tracked objects from the lost pool
        >>> lost_stracks = sub_stracks(lost_stracks, tracked_stracks)
    """
    btrack_ids = {t.track_id for t in btracks}
    return [t for t in atracks if t.track_id not in btrack_ids]


def remove_duplicate_stracks(atracks: list, btracks: list) -> tuple[list, list]:
    """Remove duplicate tracks across two lists based on Intersection over Union (IoU) distance.

    Track pairs with IoU distance < 0.15 (IoU > 0.85) are treated as duplicates of the same
    object. The shorter-lived track (smaller `frame_id - start_frame`) is dropped; ties drop
    from `atracks`.

    Args:
        atracks (list[STrack]): First list of tracks; entries must expose `xyxy`, `frame_id`,
            and `start_frame`.
        btracks (list[STrack]): Second list of tracks with the same attribute requirements.

    Returns:
        resa (list[STrack]): `atracks` with duplicate tracks removed.
        resb (list[STrack]): `btracks` with duplicate tracks removed.

    Examples:
        De-duplicate the tracked and lost pools at the end of a frame
        >>> tracked, lost = remove_duplicate_stracks(tracked_stracks, lost_stracks)
    """
    pdist = matching.iou_distance(atracks, btracks)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = [], []
    for p, q in zip(*pairs):
        timep = atracks[p].frame_id - atracks[p].start_frame
        timeq = btracks[q].frame_id - btracks[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(atracks) if i not in dupa]
    resb = [t for i, t in enumerate(btracks) if i not in dupb]
    return resa, resb


def multi_gmc(stracks: list, H: np.ndarray = np.eye(2, 3)) -> None:
    """Update multiple track positions and covariances using a 2x3 affine homography.

    The Kalman state is assumed to be laid out as `(*box, *box_velocity)` with the box
    center `(x, y)` in the first two dims. `R8x8` rotates all four 2-d pairs block-
    diagonally; the translation `t` is applied only to the position.

    Args:
        stracks (list[STrack]): Tracks to warp in place; each must expose `mean` (shape (8,))
            and `covariance` (shape (8, 8)).
        H (np.ndarray): 2x3 affine homography mapping the previous frame to the current one.

    Examples:
        Apply camera-motion compensation to the active track pool
        >>> warp = gmc.apply(frame, detection_boxes)
        >>> multi_gmc(tracked_stracks, warp)
    """
    if not stracks:
        return
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
