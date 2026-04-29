---
comments: true
description: Explore Cityscapes8, a compact 8-image semantic segmentation dataset for quickly testing Ultralytics YOLO semseg training and validation workflows.
keywords: Cityscapes8, semantic segmentation, Ultralytics YOLO, YOLO26, semseg, dataset YAML, quick test
---

# Cityscapes8 Dataset

## Introduction

Cityscapes8 is a compact semantic segmentation dataset derived from Cityscapes. It contains 8 images total, with 4 images for training and 4 images for validation. It is designed for quick testing, debugging, and CI checks before running experiments on the full [Cityscapes](cityscapes.md) dataset.

Cityscapes8 uses the same 19 Cityscapes train classes and the same `label_mapping` behavior as the full dataset.

## Dataset Structure

```text
cityscapes8/
|-- images/
|   |-- train/
|   `-- val/
`-- masks/
    |-- train/
    `-- val/
```

Each mask is a single-channel PNG file where pixel values represent class IDs. Ignored regions use pixel value `255`.

## Dataset YAML

The Cityscapes8 dataset configuration is available at [ultralytics/cfg/datasets/cityscapes8.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/cityscapes8.yaml). The YAML includes a download URL for the small packaged subset.

!!! example "ultralytics/cfg/datasets/cityscapes8.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/cityscapes8.yaml"
    ```

## Usage

Train a YOLO26 semantic segmentation model on Cityscapes8:

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n-semseg.pt")
        results = model.train(data="cityscapes8.yaml", epochs=100, imgsz=1024)
        ```

    === "CLI"

        ```bash
        yolo semseg train data=cityscapes8.yaml model=yolo26n-semseg.pt epochs=100 imgsz=1024
        ```

## FAQ

### What is Cityscapes8 used for?

Cityscapes8 is intended for fast pipeline validation. It is useful for checking that semantic segmentation training, mask loading, augmentations, validation, and export paths work before using a larger dataset.

### Should I use Cityscapes8 for benchmarking?

No. Cityscapes8 is too small for meaningful model comparison. Use the full [Cityscapes](cityscapes.md) validation set for benchmark results.
