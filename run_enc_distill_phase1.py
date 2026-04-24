#!/usr/bin/env python
"""Phase 1: Encoder distillation pretraining on DataComp-12M."""

import sys
from pathlib import Path
import os

_REPO_ROOT = str(Path(__file__).resolve().parent)
os.environ["PYTHONPATH"] = _REPO_ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")

import torch

from callbacks import paths, wandb_config
from ultralytics import YOLO
from ultralytics.models.yolo.classify.train_image_encoder import ImageEncoderTrainer

RECIPES = {
    "default": dict(lr0=3e-4, weight_decay=0.05, warmup_epochs=1, epochs=10, momentum=0.9, grad_clip=3.0, beta2=None),
    # EUPE Stage 2: proxy->student distillation (arXiv:2603.22387 Sec 4.1, ssl_default_config.yaml:131-147)
    # Same loss as ours (0.9cos+0.1L1, Eq.5-6). beta2=None -> uses default 0.999 matching EUPE
    "eupe": dict(lr0=2e-5, weight_decay=1e-4, warmup_epochs=1, epochs=30, momentum=0.9, grad_clip=3.0, beta2=None),
    # AM-RADIO: multi-teacher distillation (arXiv:2312.06709 Sec 4, Eq.2-3)
    # Same loss as ours (0.9cos+0.1L1). beta2=0.95 from MobileCLIP2 (training/configs/run_dfndr2b.sh)
    "radio": dict(lr0=1e-3, weight_decay=0.02, warmup_epochs=1, epochs=30, momentum=0.9, grad_clip=1.0, beta2=0.95),
    # UNIC (ECCV 2024) reproduction used for phase1-b1-unic-eupe-vitb16 (R1 ablation baseline).
    # lr0/wd/warmup matched from /data/shared-datasets/fatih-runs/.../phase1-b1-unic-eupe-vitb16/args.yaml.
    "unic": dict(lr0=6e-4, weight_decay=0.03, warmup_epochs=2, epochs=30, momentum=0.9, grad_clip=3.0, beta2=None),
}

# Reference global step-batch the recipes' lr0 and warmup_epochs are tuned for. When
# per_gpu_batch * world_size exceeds this, lr0 and warmup_epochs scale linearly and nbs rises
# to the global batch so wd_eff stays at the recipe value.
NBS_CANONICAL = 512


def _pop_flag(argv: list[str], flag: str, is_bool: bool = False) -> tuple[list[str], str]:
    """Pop a --flag [value] pair from argv, return (remaining_argv, value).

    Args:
        argv: argument list
        flag: flag name (e.g. "--resume")
        is_bool: if True, flag has no value argument
    """
    if flag not in argv:
        return argv, ""
    i = argv.index(flag)
    if is_bool:
        return argv[:i] + argv[i + 1 :], "true"
    return argv[:i] + argv[i + 2 :], argv[i + 1]


def _load_train_args(resume: str) -> dict:
    """Load saved training arguments from a checkpoint."""
    return torch.load(Path(resume), map_location="cpu", weights_only=False)["train_args"]


def main(argv: list[str]) -> None:
    """Launch a fresh phase 1 run or resume from a checkpoint.

    Args:
        argv: [gpu, teachers, name, recipe, model_yaml, data, epochs]
        --resume <path>: resume from checkpoint
        --cos_weight <float>: cosine loss weight (default 0.9)
        --l1_weight <float>: smooth L1 loss weight (default 0.1)
        --cls_l1: add smooth L1 to CLS token loss (default False)
        --lr <float>: override recipe lr0 (applied before batch scaling)
        --batch <int>: per-GPU (per-rank) batch. Global batch = per-GPU * world_size. When the
            global batch exceeds NBS_CANONICAL (512), lr0 and warmup_epochs scale linearly and
            nbs is raised to the global batch so wd_eff is invariant.
    """
    args = argv[1:]
    args, resume = _pop_flag(args, "--resume")
    args, cos_w = _pop_flag(args, "--cos_weight")
    args, l1_w = _pop_flag(args, "--l1_weight")
    args, cls_l1_str = _pop_flag(args, "--cls_l1", is_bool=True)
    args, lr_override = _pop_flag(args, "--lr")
    args, batch_override = _pop_flag(args, "--batch")
    args, fork_from = _pop_flag(args, "--fork_from")  # format: <parent_run_id>:<fork_step>
    args, distill_path = _pop_flag(args, "--distill_path")
    args, adaptor_arch = _pop_flag(args, "--adaptor_arch")

    cos_weight = float(cos_w) if cos_w else 0.9
    l1_weight = float(l1_w) if l1_w else 0.1
    cls_l1 = bool(cls_l1_str)
    distill_path = distill_path or "adaptor"
    adaptor_arch = adaptor_arch or "mlp"

    if resume:
        resume = paths.patch_resume(resume)
    resume_args = _load_train_args(resume) if resume else {}

    # Resume guard: prevent silent distill_path / adaptor_arch switches. These change graph topology
    # (feat_adaptors built or not) and loss_items labels; switching mid-run corrupts checkpoints and WandB plots.
    if resume_args:
        for key, now in (("distill_path", distill_path), ("adaptor_arch", adaptor_arch)):
            prev = resume_args.get(key, "adaptor" if key == "distill_path" else "mlp")
            if now != prev:
                raise ValueError(
                    f"Refusing resume: --{key} mismatch (ckpt={prev!r} vs cli={now!r}). "
                    f"Either drop the flag or start a fresh run."
                )
    gpu = args[0] if args else "0"
    teachers = args[1] if len(args) > 1 else resume_args.get("teachers", "eupe:vitb16")
    name = (
        args[2] if len(args) > 2 else resume_args.get("name", f"phase1-{teachers.replace(':', '-').replace('+', '_')}")
    )
    recipe = args[3] if len(args) > 3 else "default"
    model_yaml = args[4] if len(args) > 4 else "yolo26s-cls.yaml"
    data = args[5] if len(args) > 5 else "/data/shared-datasets/datacomp-12m"
    epochs = int(args[6]) if len(args) > 6 else None
    r = RECIPES[recipe]

    world_size = len(gpu.split(",")) if "," in gpu else 1
    global_batch = int(batch_override or 64) * world_size  # default per-GPU = 64 (anchor per-rank)
    scale = max(1.0, global_batch / NBS_CANONICAL)
    lr0 = float(lr_override or r["lr0"]) * scale
    nbs = max(global_batch, NBS_CANONICAL)
    warmup_epochs = r["warmup_epochs"] * scale

    model = YOLO(model_yaml)
    # grad_clip, beta2, nfs_sync registered inside ImageEncoderTrainer (survives DDP respawn).
    model.add_callback(
        "on_pretrain_routine_start",
        wandb_config.log_config(
            model=model_yaml,
            teachers=teachers,
            recipe=recipe,
            cos_weight=cos_weight,
            l1_weight=l1_weight,
            cls_l1=cls_l1,
            distill_path=distill_path,
            adaptor_arch=adaptor_arch,
            grad_clip=r["grad_clip"],
            beta2=r["beta2"],
            wandb_group="distill",
        ),
    )
    train_args = dict(
        trainer=ImageEncoderTrainer,
        teachers=teachers,
        data=data,
        knn_eval="/data/shared-datasets/imagenet",
        cos_weight=cos_weight,
        l1_weight=l1_weight,
        cls_l1=cls_l1,
        distill_path=distill_path,
        adaptor_arch=adaptor_arch,
        device=gpu,
        **paths.run_paths(name),
        epochs=epochs or r["epochs"],
        batch=global_batch,
        imgsz=224,
        patience=5,
        nbs=nbs,
        cos_lr=True,
        lr0=lr0,
        lrf=0.01,
        momentum=r["momentum"],
        weight_decay=r["weight_decay"],
        grad_clip=r["grad_clip"],
        beta2=r["beta2"],
        warmup_epochs=warmup_epochs,
        warmup_bias_lr=0,
        dropout=0,
        optimizer="AdamW",
        pretrained=False,
        amp=True,
        seed=0,
        deterministic=True,
        fliplr=0.5,
        workers=2,
        nfs_sync=True,
    )
    if resume:
        train_args["resume"] = resume
    if fork_from:
        parent_id, fork_step = fork_from.split(":")
        wandb_config.fork_and_attach(parent_id, int(fork_step), name)
    model.train(**train_args)


if __name__ == "__main__":
    main(sys.argv)
