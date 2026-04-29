---
comments: true
description: Learn how to prepare semantic segmentation datasets for Ultralytics YOLO, including PNG mask labels, dataset YAML fields, ignore labels, and supported datasets.
keywords: Ultralytics, YOLO, semantic segmentation, semseg, dataset format, pixel masks, Cityscapes, ADE20K, Pascal VOC
---

# Semantic Segmentation Datasets Overview

[Semantic segmentation](https://www.ultralytics.com/glossary/semantic-segmentation) assigns one class label to every pixel in an image. Unlike [instance segmentation](../segment/index.md), semantic segmentation does not separate individual objects of the same class. The training target is a dense class map where each pixel stores a class ID.

This guide explains the dataset format used by Ultralytics YOLO semantic segmentation models and lists the built-in dataset configurations available for training and validation.

## Supported Dataset Format

### PNG mask format

Semantic segmentation datasets use one image file and one mask file per sample. The mask is a single-channel image, usually PNG, where each pixel value is the class index for the corresponding image pixel.

- Pixel values `0`, `1`, `2`, ... represent class IDs from the dataset `names` mapping.
- Pixel value `255` is treated as the ignore label and is excluded from loss and metric computation.
- Mask files should use the same stem as their matching image file, for example `frankfurt_000000_000294.png`.
- Supported mask extensions are `.png`, `.PNG`, `.bmp`, and `.tif`.

### Directory layout

The default layout keeps images and masks in parallel folders. The `masks_dir` value from the dataset YAML replaces the `images` path component to find masks.

```text
dataset/
|-- images/
|   |-- train/
|   `-- val/
`-- masks/
    |-- train/
    `-- val/
```

For example, an image at `images/train/aachen_000000_000019.png` is paired with a mask at `masks/train/aachen_000000_000019.png` when `masks_dir: masks`.

### Dataset YAML format

Semantic segmentation datasets are configured with YAML files. The main fields are:

| Key             | Description                                                                 |
| --------------- | --------------------------------------------------------------------------- |
| `path`          | Dataset root directory.                                                     |
| `train`         | Training image path relative to `path`, or an absolute path.                |
| `val`           | Validation image path relative to `path`, or an absolute path.              |
| `test`          | Optional test image path.                                                   |
| `masks_dir`     | Directory name used for semantic masks. Defaults to `masks` when omitted.   |
| `names`         | Class ID to class name mapping.                                             |
| `label_mapping` | Optional mapping from source dataset IDs to training IDs or `ignore_label`. |

!!! example "ultralytics/cfg/datasets/cityscapes8.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/cityscapes8.yaml"
    ```

Use `label_mapping` when the source mask IDs do not already match contiguous training class IDs. Cityscapes and ADE20K include mappings that convert original label IDs into YOLO semantic segmentation train IDs and ignore unused labels.

## Usage

Train a YOLO26 semantic segmentation model with Python or CLI:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained semantic segmentation model
        model = YOLO("yolo26n-semseg.pt")

        # Train on the Cityscapes8 semantic segmentation dataset
        results = model.train(data="cityscapes8.yaml", epochs=100, imgsz=1024)
        ```

    === "CLI"

        ```bash
        yolo semseg train data=cityscapes8.yaml model=yolo26n-semseg.pt epochs=100 imgsz=1024
        ```

## Supported Datasets

Ultralytics provides semantic segmentation dataset YAML files for these datasets:

- [Cityscapes](cityscapes.md): Urban street-scene semantic segmentation dataset with 19 train classes.
- [Cityscapes8](cityscapes8.md): An 8-image Cityscapes subset for quick tests and CI checks.
- [ADE20K](ade20k.md): Scene parsing dataset with 150 semantic classes.

## Adding Your Own Dataset

To train on a custom semantic segmentation dataset:

1. Save your images under split folders such as `images/train` and `images/val`.
2. Save one single-channel mask per image under the mirrored mask folders, such as `masks/train` and `masks/val`.
3. Ensure mask pixel values are class IDs. Use `255` for pixels that should be ignored.
4. Create a dataset YAML with `path`, `train`, `val`, `masks_dir`, and `names`.
5. Add `label_mapping` only when your mask IDs need conversion to contiguous train IDs.

```yaml
path: path/to/my-semseg-dataset
train: images/train
val: images/val
masks_dir: masks

names:
  0: background
  1: road
  2: building
```

## FAQ

### What is the difference between semantic segmentation masks and instance segmentation labels?

Semantic segmentation masks are dense pixel maps. Each pixel stores a class ID, and there is one mask image per training image. Instance segmentation labels in Ultralytics YOLO use text files with polygon coordinates, one row per object instance.

### What pixel value is ignored during training?

Pixel value `255` is used as the ignore label. These pixels are skipped during loss and metric computation, which is useful for void regions, unlabeled pixels, or classes outside the training label set.

### Do mask file names need to match image file names?

Yes. Each semantic mask should have the same file stem as the corresponding image. The dataset loader replaces the `images` directory component with `masks_dir` and searches for matching mask files.

### Can I use original dataset label IDs directly?

Yes, if they already match your `names` class IDs. If the source dataset uses non-contiguous IDs or includes labels that should be ignored, add a `label_mapping` section to convert source pixel values to training IDs.
