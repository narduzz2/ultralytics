---
description: Explore the Deep OC-SORT tracker in Ultralytics — OC-SORT extended with adaptive ReID appearance fusion, dynamic appearance EMA, and global motion compensation (GMC) for robust multi-object tracking under occlusion and camera motion.
keywords: Ultralytics, Deep OC-SORT, DeepOCSORT, DeepOCSortTrack, ReID, appearance, GMC, observation-centric, object tracking, MOT
---

# Reference for `ultralytics/trackers/deep_oc_sort.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/deep_oc_sort.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/deep_oc_sort.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

Deep OC-SORT (Maggiolino et al., [arXiv:2302.11813](https://arxiv.org/abs/2302.11813)) builds on [OC-SORT](oc_sort.md) by injecting deep appearance cues into the association: detection embeddings are fused into the cost matrix only where they are reliable (via **Adaptive Weighting** and **Dynamic Appearance** EMA), and **Camera Motion Compensation** corrects Kalman states for ego-motion. Enable it with `tracker="deepocsort.yaml"`; the default config ships with a YOLO-native ReID model, but `model: auto` reuses detector features when supported.

<br>

## ::: ultralytics.trackers.deep_oc_sort.DeepOCSortTrack

<br><br><hr><br>

## ::: ultralytics.trackers.deep_oc_sort.DeepOCSORT

<br><br>
