---
comments: true
description: Per-image property correlation analysis for object detection. Surface which image properties drive bad model performance, rank worst-performing images, and feed them into a synthetic-data-generation loop.
keywords: Ultralytics, image property analysis, label quality, ObjectLab, correlation, worst images, synthetic data, data-centric
---

# Image Property Correlation Analysis

After training a detection model, you often want to know **why** some images are mispredicted. The [`ImagePropertyAnalyzer`](../reference/utils/analysis.md) joins per-image F1 scores from validation with image properties (brightness, blurriness, crowdedness, label-quality scores, etc.), computes Pearson and Spearman correlations against F1, and ranks the worst-performing images so you can feed them into a synthetic-data pipeline.

The analyzer ships **31 properties** out of the box: 8 pixel-reading (brightness, contrast, entropy, edge density, ...), 17 cache-derived (object counts in COCO size buckets, class entropy, edge proximity, ...), 2 annotation-interaction (max/mean pairwise IoU), and 4 ObjectLab label-quality scores (overlooked, badloc, swap, label_quality_score) following [Tkachenko, Thyagarajan & Mueller, ICML Workshop 2023](https://arxiv.org/abs/2309.00832).

## Quick start

Three entry-point paths cover the common use cases:

```python
from ultralytics import YOLO
from ultralytics.utils.analysis import ImagePropertyAnalyzer

# Path 1: model + dataset, runs validation internally
report = ImagePropertyAnalyzer(model="yolo11n.pt", data="coco128.yaml").run()

# Path 2: from a previous model.val() result, no re-validation
m = YOLO("yolo11n.pt")
metrics = m.val(data="coco128.yaml", analyze_images=True)
report = ImagePropertyAnalyzer.from_metrics(metrics, dataset=m.validator.dataloader.dataset).run()

# Path 3: dataset-only audit, no model required
report = ImagePropertyAnalyzer(data="coco128.yaml").run()
```

Each call writes the following to a timestamped `runs/analysis/` directory:

| File                      | Purpose                                                                               |
| ------------------------- | ------------------------------------------------------------------------------------- |
| `per_image_analysis.csv`  | One row per image, sorted ascending by F1 (or by anomaly_score in dataset-only path)  |
| `correlations.json`       | Pearson + Spearman r and p-values per property, with effect-size band and direction   |
| `worst_images.json`       | Top 100 worst-performing images plus their top 3 problematic properties               |
| `summary.md`              | Human-readable summary with top correlations and worst-image table                    |
| `correlation_scatter.png` | Per-property scatter against F1 with regression line and Pearson r                    |
| `correlation_heatmap.png` | Property × property Pearson r matrix (self-correlations blanked)                      |
| `worst_images_strip.png`  | Thumbnails of bottom 20 by F1 with green ground-truth and red dashed prediction boxes |

## Example outputs

Rendered on COCO val2017 (5000 images) with `yolo11n.pt` at `conf=0.25`. `summary.md` reports the strongest correlates (object count, object size variation, small-object count) in plain English and links the three plots:

![F1 vs each property](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/image-property-correlation-scatter.avif)

![Property correlation heatmap](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/image-property-correlation-heatmap.avif)

![Worst 20 images](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/image-property-worst-images-strip.avif)

## Enabling label-quality scores

The 4 ObjectLab fields (`overlooked_score`, `badloc_score`, `swap_score`, `label_quality_score`) require the validator to compute them inline during validation. Pass `analyze_images=True` to `model.val()`:

```python
metrics = m.val(data="coco128.yaml", analyze_images=True)
```

The model+data path (`ImagePropertyAnalyzer(model=..., data=...)`) sets this flag automatically. The validator stores ~32 bytes/image extra in `metrics.box.image_metrics` (4 float scores per image). Raw IoU matrices and pred/GT arrays are not retained. Without the flag, ObjectLab columns are populated as `NaN`.

All 4 ObjectLab scores follow the quality convention: **low = likely label issue**, **high = clean label**.

## Platform integration (`ul://`)

Both the `model=` and `data=` arguments accept the modern Ultralytics Platform URI scheme. The analyzer forwards them to existing `YOLO()` and `check_det_dataset()` resolution paths, which handle download and conversion:

```python
ImagePropertyAnalyzer(
    model="ul://owner/project/model-name",
    data="ul://owner/datasets/slug",
    api_key="ul_xxx_40hex",  # optional, falls back to ULTRALYTICS_API_KEY env or settings
).run()
```

See the [Platform API docs](https://docs.ultralytics.com/platform/api/) for URI details.

## Property catalog and references

| Feature / per-image field                                                             | Source                                                                                                                                            |
| ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `brightness` (HSP perceptual)                                                         | [Hendrycks & Dietterich, ICLR 2019](https://arxiv.org/abs/1903.12261)                                                                             |
| `dark_pixel_ratio`, `bright_pixel_ratio`                                              | [Hendrycks & Dietterich, ICLR 2019](https://arxiv.org/abs/1903.12261)                                                                             |
| `blurriness` (variance-of-Laplacian)                                                  | [Pech-Pacheco et al., ICPR 2000](https://doi.org/10.1109/ICPR.2000.903548)                                                                        |
| `entropy` (Shannon over grayscale histogram)                                          | [Shannon, BSTJ 1948](https://doi.org/10.1002/j.1538-7305.1948.tb01338.x)                                                                          |
| `contrast` (grayscale std)                                                            | [Hendrycks & Dietterich, ICLR 2019](https://arxiv.org/abs/1903.12261)                                                                             |
| `edge_density` (Canny edge mean)                                                      | [Canny, IEEE TPAMI 1986](https://doi.org/10.1109/TPAMI.1986.4767851)                                                                              |
| `sharpness` (Tenengrad gradient)                                                      | [Krotkov, IJCV 1988](https://doi.org/10.1007/BF00127822)                                                                                          |
| `aspect_ratio`, `width`, `height`, `total_pixels`, `num_objects`                      | trivial                                                                                                                                           |
| `num_small` / `num_medium` / `num_large` (COCO area buckets 32², 96²)                 | [Lin et al., COCO, ECCV 2014](https://arxiv.org/abs/1405.0312)                                                                                    |
| `small_object_ratio`, `box_area_std_norm`, `object_scale_variance`                    | trivial                                                                                                                                           |
| `num_classes_present`                                                                 | trivial                                                                                                                                           |
| `class_entropy`                                                                       | [Shannon, BSTJ 1948](https://doi.org/10.1002/j.1538-7305.1948.tb01338.x)                                                                          |
| `mean_center_x`, `mean_center_y`, `center_spread`                                     | trivial                                                                                                                                           |
| `num_near_edge` (boundary-truncated objects)                                          | [Everingham et al., Pascal VOC, IJCV 2010](https://link.springer.com/article/10.1007/s11263-009-0275-4)                                           |
| `max_pairwise_iou`, `mean_pairwise_iou` (per-image crowdedness)                       | [Shao et al., CrowdHuman, 2018](https://arxiv.org/abs/1805.00123)                                                                                 |
| `overlooked_score`, `badloc_score`, `swap_score`, `label_quality_score` (ObjectLab)   | [Tkachenko, Thyagarajan & Mueller, ICML Workshop 2023](https://arxiv.org/abs/2309.00832)                                                          |
| Per-image P/R/F1/TP/FP/FN                                                             | in-tree validator                                                                                                                                 |
| Pearson + Spearman correlation per property × F1 with effect-size band                | [Pearson, Proc. Royal Society 1895](https://doi.org/10.1098/rspl.1895.0041) / [Spearman, Am. J. Psychology 1904](https://doi.org/10.2307/1412159) |
| Worst-image ranking + scatter grid + heatmap + worst-image strip plots + `summary.md` | in-tree                                                                                                                                           |
| `ul://` platform-URI resolution for model + dataset inputs                            | [Ultralytics Platform API docs](https://docs.ultralytics.com/platform/api/)                                                                       |

## Output schema

`per_image_analysis.csv` columns: `im_name`, `im_path`, then validator-supplied prediction-quality fields (`precision`, `recall`, `f1`, `tp`, `fp`, `fn`) when predictions are available, then all 31 property fields plus `anomaly_score`. The CSV is always fully sorted, ascending by F1 (model+data and from-metrics paths) or descending by `anomaly_score` (dataset-only path).

`correlations.json` entries:

```json
{
    "brightness": {
        "pearson_r": -0.34,
        "pearson_p": 1.2e-5,
        "spearman_r": -0.31,
        "spearman_p": 3.4e-5,
        "n": 458,
        "effect_band": "moderate",
        "direction": "higher brightness -> lower F1"
    }
}
```

`worst_images.json` entries:

```json
[
    {
        "im_name": "img_0042.jpg",
        "f1": 0.12,
        "anomaly_score": 2.31,
        "top_3_problematic": ["blurriness", "num_small", "num_near_edge"]
    }
]
```

## Acting on the results

The report surfaces _which_ image properties drive low per-image F1. Common follow-ups:

- **Crowdedness / object count**: if `num_objects`, `max_pairwise_iou`, or `small_object_ratio` correlate with low F1, your model struggles in dense scenes. Consider raising `imgsz`, training with more crowded-scene augmentation (mosaic, copy-paste), or generating synthetic crowded scenes targeting the worst images.
- **Object scale spread**: if `object_scale_variance` or `num_small` correlate with low F1, multi-scale predictions are weak. Tune anchor-free head capacity or add tiled-inference for small targets.
- **Pixel-level corruptions**: brightness/contrast/blurriness/`dark_pixel_ratio` correlations point at exposure or motion-blur issues. Augment with the corresponding [Albumentations](../integrations/albumentations.md) transforms, or retrain after curating examples with similar properties.
- **Label-quality scores** (`overlooked_score`, `badloc_score`, `swap_score`, `label_quality_score`): low scores flag specific annotation issues per image. Review the listed worst images, fix labels, and retrain.
- **Worst-image triage**: the listed worst images are direct candidates for synthetic-data targets: generate variants with the highlighted properties amplified, label them, and add to the training set.

The `anomaly_score` per image is a signed z-score average across all properties, weighted so positive = unusual in an F1-degrading direction. Treat large positive values as "this image is statistically the kind of input your model struggles with."

## Caveats

- **Filename collisions**: `Metric.image_metrics` is keyed by image basename. If your dataset has duplicate basenames across subdirectories they collide silently. The analyzer emits a single `LOGGER.warning` listing the count and a few examples.
- **Empty-label images**: zero-box images break per-image-box stats (mean undefined). The analyzer emits `NaN` for those properties and excludes them from correlations.
- **Tasks supported**: 27 image-property fields work for any of detection / segmentation / pose / OBB. The 4 ObjectLab fields ship for **detection only** (segmentation, pose, and OBB extensions via mask-IoU, OKS, and rotated-box similarity are deferred to a follow-up release).
- **DDP**: the validator-side retention path is rank-0 safe, the existing `dist.gather_object` plumbing pickles numpy arrays cleanly without new logic.
