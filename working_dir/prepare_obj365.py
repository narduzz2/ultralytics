"""Extract Objects365 patches and generate YOLO labels from COCO JSON.

Replicates the Objects365.yaml download block without the download step.
Assumes patches are already in images/train/ and images/val/, and
zhiyuan_objv2_train.json / zhiyuan_objv2_val.json are at the dataset root.

Usage:
    python working_dir/prepare_obj365.py --dataset /data/datasets/Objects365
    python working_dir/prepare_obj365.py --dataset /data/datasets/Objects365 --skip-extract
    python working_dir/prepare_obj365.py --dataset /data/datasets/Objects365 --skip-labels
"""

from __future__ import annotations

import argparse
import tarfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from ultralytics.utils import LOGGER, TQDM
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.ops import xyxy2xywhn


def _extract_patch(args: tuple) -> str:
    """Extract one patch*.tar.gz into its parent directory."""
    tar_path, images_dir = args
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(images_dir, set_attrs=False)
    return Path(tar_path).name


def extract_and_flatten(images_dir: Path, workers: int) -> None:
    """Extract all patch*.tar.gz in images_dir then flatten .jpg files into it."""
    tarballs = sorted(images_dir.glob("patch*.tar.gz"))
    if not tarballs:
        LOGGER.info(f"  No patch*.tar.gz found in {images_dir}")
        return

    LOGGER.info(f"  Extracting {len(tarballs)} patches ...")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_extract_patch, (str(t), str(images_dir))): t.name for t in tarballs}
        for fut in as_completed(futures):
            LOGGER.info(f"    {futures[fut]} done")

    # Move all nested .jpg to images_dir (same as Objects365.yaml)
    files = list(images_dir.rglob("*.jpg"))
    nested = [f for f in files if f.parent != images_dir]
    LOGGER.info(f"  Moving {len(nested)} images ...")
    with ThreadPoolExecutor(max_workers=16) as ex:
        list(TQDM(ex.map(lambda f: f.rename(images_dir / f.name), nested), total=len(nested), desc="Moving"))

    for d in sorted(images_dir.rglob("*"), reverse=True):
        if d.is_dir():
            try:
                d.rmdir()
            except OSError:
                pass


def generate_labels(json_path: Path, labels_dir: Path) -> None:
    """Generate YOLO label .txt files from a COCO JSON — same logic as Objects365.yaml."""
    check_requirements("faster-coco-eval")
    from faster_coco_eval import COCO

    labels_dir.mkdir(parents=True, exist_ok=True)
    coco = COCO(json_path)
    names = [x["name"] for x in coco.loadCats(coco.getCatIds())]

    for cid, cat in enumerate(names):
        cat_ids = coco.getCatIds(catNms=[cat])
        img_ids = coco.getImgIds(catIds=cat_ids)

        def process_annotation(im):
            width, height = im["width"], im["height"]
            path = Path(im["file_name"])
            with open(labels_dir / path.with_suffix(".txt").name, "a", encoding="utf-8") as file:
                ann_ids = coco.getAnnIds(imgIds=im["id"], catIds=cat_ids, iscrowd=None)
                for a in coco.loadAnns(ann_ids):
                    x, y, w, h = a["bbox"]
                    xyxy = np.array([x, y, x + w, y + h])[None]
                    x, y, w, h = xyxy2xywhn(xyxy, w=width, h=height, clip=True)[0]
                    file.write(f"{cid} {x:.5f} {y:.5f} {w:.5f} {h:.5f}\n")

        imgs = coco.loadImgs(img_ids)
        with ThreadPoolExecutor(max_workers=16) as ex:
            list(TQDM(ex.map(process_annotation, imgs), total=len(imgs), desc=f"Class {cid + 1}/{len(names)} {cat}"))


def main():
    parser = argparse.ArgumentParser(description="Extract Objects365 patches and generate YOLO labels")
    parser.add_argument("--dataset", required=True, type=Path, help="Objects365 dataset root")
    parser.add_argument("--skip-extract", action="store_true", help="Skip tar.gz extraction")
    parser.add_argument("--skip-labels", action="store_true", help="Skip label generation")
    parser.add_argument("--workers", type=int, default=8, help="Workers for parallel extraction (default: 8)")
    args = parser.parse_args()

    root = args.dataset

    if not args.skip_extract:
        for split in ("train", "val"):
            images_dir = root / "images" / split
            if not images_dir.exists():
                LOGGER.warning(f"  {images_dir} not found, skipping.")
                continue
            LOGGER.info(f"\nExtracting {split} images ...")
            extract_and_flatten(images_dir, args.workers)

    if not args.skip_labels:
        for split in ("train", "val"):
            json_path = root / f"zhiyuan_objv2_{split}.json"
            if not json_path.exists():
                LOGGER.warning(f"  {json_path} not found, skipping {split} labels.")
                continue
            LOGGER.info(f"\nGenerating {split} labels ...")
            generate_labels(json_path, root / "labels" / split)

    LOGGER.info(f"\nDone. Dataset ready at {root}")


if __name__ == "__main__":
    main()
