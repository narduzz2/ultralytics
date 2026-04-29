---
comments: true
description: Learn how to use the ADE20K semantic segmentation dataset with Ultralytics YOLO, including scene parsing masks, YAML configuration, and training examples.
keywords: ADE20K, semantic segmentation, scene parsing, Ultralytics YOLO, YOLO26, ADEChallengeData2016, dataset YAML
---

# ADE20K Dataset

## Introduction

[ADE20K](http://sceneparsing.csail.mit.edu/) is a scene parsing dataset for semantic segmentation. The Ultralytics `ade20k.yaml` configuration uses 150 semantic classes and maps the original ADE20K background or ignore value to `255`.

ADE20K is useful for training models that need dense scene understanding across indoor, outdoor, object, and stuff categories.

## Dataset Structure

The Ultralytics configuration expects the ADEChallengeData2016 layout:

```text
ADEChallengeData2016/
|-- images/
|   |-- training/
|   `-- validation/
`-- annotations/
    |-- training/
    `-- validation/
```

The `masks_dir` field is set to `annotations`, so image paths under `images/` are paired with masks under `annotations/`.

## Dataset YAML

The ADE20K dataset configuration is available at [ultralytics/cfg/datasets/ade20k.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ade20k.yaml).

!!! example "ultralytics/cfg/datasets/ade20k.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/ade20k.yaml"
    ```

## Usage

Train a YOLO26 semantic segmentation model on ADE20K:

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n-semseg.pt")
        results = model.train(data="ade20k.yaml", epochs=100, imgsz=512)
        ```

    === "CLI"

        ```bash
        yolo semseg train data=ade20k.yaml model=yolo26n-semseg.pt epochs=100 imgsz=512
        ```

## Citations and Acknowledgments

If you use ADE20K in your research, please cite the ADE20K paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @inproceedings{zhou2017scene,
          title={Scene Parsing through ADE20K Dataset},
          author={Zhou, Bolei and Zhao, Hang and Puig, Xavier and Fidler, Sanja and Barriuso, Adela and Torralba, Antonio},
          booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
          year={2017}
        }
        ```

## FAQ

### How many classes does ADE20K use?

The Ultralytics ADE20K configuration uses 150 semantic segmentation classes.

### Why does ADE20K use `label_mapping`?

ADE20K annotation masks use source label IDs where `0` is ignored. The `label_mapping` section maps valid labels `1` through `150` to contiguous train IDs `0` through `149` and maps ignored pixels to `255`.
