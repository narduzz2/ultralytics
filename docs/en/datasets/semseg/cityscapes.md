---
comments: true
description: Learn how to use the Cityscapes semantic segmentation dataset with Ultralytics YOLO, including dataset structure, YAML configuration, and training examples.
keywords: Cityscapes, semantic segmentation, Ultralytics YOLO, YOLO26, autonomous driving, urban scenes, dataset YAML
---

# Cityscapes Dataset

## Introduction

[Cityscapes](https://www.cityscapes-dataset.com/) is a semantic segmentation dataset focused on urban street scenes. It is widely used for autonomous driving and scene understanding research. The Ultralytics `cityscapes.yaml` configuration uses the standard 19 semantic segmentation train classes.

Cityscapes fine annotations include 2,975 training images, 500 validation images, and 1,525 test images. The Ultralytics YAML expects the official `leftImg8bit` and `gtFine` archives to be downloaded from Cityscapes and extracted into the dataset root before the processing script organizes images and masks.

## Dataset Structure

After preparation, the dataset is organized as:

```text
cityscapes/
|-- images/
|   |-- train/
|   |-- val/
|   `-- test/
`-- masks/
    |-- train/
    |-- val/
    `-- test/
```

The semantic masks are single-channel PNG files. Original Cityscapes label IDs are mapped to 19 train IDs with `label_mapping`; ignored and void labels are mapped to `255`.

## Dataset YAML

The Cityscapes dataset configuration is available at [ultralytics/cfg/datasets/cityscapes.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/cityscapes.yaml).

!!! example "ultralytics/cfg/datasets/cityscapes.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/cityscapes.yaml"
    ```

## Usage

Train a YOLO26 semantic segmentation model on Cityscapes:

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n-semseg.pt")
        results = model.train(data="cityscapes.yaml", epochs=100, imgsz=1024)
        ```

    === "CLI"

        ```bash
        yolo semseg train data=cityscapes.yaml model=yolo26n-semseg.pt epochs=100 imgsz=1024
        ```

Validate a trained model:

!!! example "Validation Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n-semseg.pt")
        metrics = model.val(data="cityscapes.yaml", imgsz=1024)
        print(metrics.miou)
        ```

    === "CLI"

        ```bash
        yolo semseg val data=cityscapes.yaml model=yolo26n-semseg.pt imgsz=1024
        ```

## Citations and Acknowledgments

If you use Cityscapes in your research, please cite the Cityscapes paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @inproceedings{Cordts2016Cityscapes,
          title={The Cityscapes Dataset for Semantic Urban Scene Understanding},
          author={Cordts, Marius and Omran, Mohamed and Ramos, Sebastian and Rehfeld, Timo and Enzweiler, Markus and Benenson, Rodrigo and Franke, Uwe and Roth, Stefan and Schiele, Bernt},
          booktitle={Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
          year={2016}
        }
        ```

## FAQ

### Why does Cityscapes use `label_mapping`?

Cityscapes source masks use original label IDs that are not the same as the 19 train IDs. The `label_mapping` section converts valid labels to contiguous class IDs and maps ignored labels to `255`.

### Do I need to download Cityscapes manually?

Yes. Cityscapes requires accepting the dataset terms on the official website. Download and extract `leftImg8bit` and `gtFine` into the `cityscapes` dataset root before running the Ultralytics preparation script.
