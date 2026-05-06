#!/usr/bin/env python
"""CE-on-ImageNet pretrain matching exp5b-ce-baseline recipe (single-GPU only).

Recipe: byte-exact match to exp5b-ce-baseline (encoder-distillation.md line 273:
"batch=256, nbs=256, MuSGD lr=0.1, muon_w=0.1, warmup_epochs=0, cos_lr=True"),
with epochs compressed 200 -> 114 to match the phase-1 distill epoch budget.

Single-GPU only: this runner registers muon_w / wandb_config via
model.add_callback(), which is silently dropped on DDP respawn per
CLAUDE.md "Phase-2 DDP callback loss". For multi-GPU, subclass
ClassificationTrainer and register callbacks inside __init__.

Usage:
    python run_cls_imagenet.py <gpu> <model_yaml> <name>
"""
import os
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent)
os.environ["PYTHONPATH"] = _REPO_ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")

from callbacks import muon_w, paths, wandb_config
from ultralytics import YOLO


def main(argv: list[str]) -> None:
    gpu = argv[1]
    model_yaml = argv[2]
    name = argv[3]
    if "," in gpu:
        raise ValueError(
            f"Single-GPU only (gpu={gpu!r}). Callbacks added via model.add_callback() "
            f"silently no-op under DDP. For multi-GPU, subclass ClassificationTrainer."
        )

    model = YOLO(model_yaml)
    model.add_callback("on_train_start", muon_w.override(0.1))
    model.add_callback(
        "on_pretrain_routine_start",
        wandb_config.log_config(
            model=model_yaml,
            recipe="exp5b-114ep",
            muon_w=0.1,
            wandb_group="ce-pretrain-imagenet",
        ),
    )
    model.train(
        data="/data/shared-datasets/imagenet",
        epochs=114,
        patience=100,
        batch=256,
        imgsz=224,
        workers=2,
        pretrained=False,
        optimizer="MuSGD",
        seed=0,
        deterministic=True,
        cos_lr=True,
        close_mosaic=10,
        amp=True,
        lr0=0.1,
        lrf=0.01,
        momentum=0.9,
        weight_decay=0.0001,
        warmup_epochs=0,
        warmup_momentum=0.8,
        warmup_bias_lr=0,
        nbs=256,
        mosaic=1,
        auto_augment="randaugment",
        erasing=0.4,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.4,
        fliplr=0.5,
        device=gpu,
        **paths.run_paths(name),
    )


if __name__ == "__main__":
    main(sys.argv)
