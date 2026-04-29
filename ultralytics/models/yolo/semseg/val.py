# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image

from ultralytics.data.build import build_dataloader
from ultralytics.data.dataset import SemsegDataset
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.metrics import SemsegMetrics


class SemanticSegmentationValidator(BaseValidator):
    """Validator for semantic segmentation models.

    This validator evaluates semantic segmentation models using mIoU and pixel accuracy metrics.

    Attributes:
        metrics (SemanticMetrics): Metrics calculator for semantic segmentation.

    Examples:
        >>> from ultralytics.models.yolo.semseg import SemanticSegmentationValidator
        >>> args = dict(model="yolo26n-semseg.pt", data="cityscapes8.yaml")
        >>> validator = SemanticSegmentationValidator(args=args)
        >>> validator()
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        """Initialize SemanticSegmentationValidator.

        Args:
            dataloader (DataLoader, optional): DataLoader for validation.
            save_dir (Path, optional): Directory to save results.
            args (dict, optional): Arguments for the validator.
            _callbacks (list, optional): Callback functions.
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "semseg"
        self.metrics = SemsegMetrics()
        self.dataset = None
        self.results_dir = None
        self.image_shapes = {}
        self._semantic_target_shape = None

    def init_metrics(self, model):
        """Initialize metrics with model class names.

        Args:
            model (nn.Module): Model to validate.
        """
        self.names = model.names
        self.nc = len(self.names)
        self.metrics = SemsegMetrics(names=self.names, device=self.device)
        self.dataset = getattr(self.dataloader, "dataset", None)
        labels = getattr(self.dataset, "labels", []) if self.dataset is not None else []
        self.image_shapes = {lb["im_file"]: tuple(lb["shape"]) for lb in labels if "im_file" in lb and "shape" in lb}
        self.results_dir = None
        if self.args.save_json:
            self.results_dir = self.save_dir / "results"
            self.results_dir.mkdir(parents=True, exist_ok=True)

    def preprocess(self, batch):
        """Preprocess a batch of images and masks.

        Args:
            batch (dict): Batch data containing images and masks.

        Returns:
            (dict): Preprocessed batch.
        """
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
        batch["semantic_mask"] = batch["semantic_mask"].to(self.device, non_blocking=True).long()
        self._semantic_target_shape = tuple(batch["semantic_mask"].shape[-2:])
        return batch

    def postprocess(self, preds):
        """Convert logits to class predictions.

        Args:
            preds (torch.Tensor): Raw model output logits [B, nc, H, W].

        Returns:
            (torch.Tensor): Predicted class IDs [B, H, W].
        """
        if isinstance(preds, (tuple, list)):
            preds = preds[0]
        preds = (
            F.interpolate(preds, size=self._semantic_target_shape, mode="bilinear", align_corners=False)
            if self._semantic_target_shape is not None
            else F.interpolate(preds, scale_factor=4, mode="bilinear", align_corners=False)
        )
        if self.nc > 1:
            pred_mask = preds.argmax(dim=1)
        else:
            pred_mask = preds.gt(0).squeeze(1)
        return pred_mask

    def update_metrics(self, preds, batch):
        """Update metrics with predictions and ground truth.

        Args:
            preds (torch.Tensor): Predicted class IDs [B, H, W].
            batch (dict): Batch containing 'semantic_mask'.
        """
        targets = batch["semantic_mask"]
        if preds.shape[1:] != targets.shape[1:]:
            preds = F.interpolate(preds.float().unsqueeze(1), targets.shape[1:], mode="nearest").squeeze(1).long()
        if self.args.save_json:
            self.save_pred_masks(preds, batch)
        self.metrics.process(preds, targets)

    def finalize_metrics(self):
        """Set final values on semantic metrics."""
        self.metrics.speed = self.speed
        self.metrics.save_dir = self.save_dir

    def gather_stats(self):
        """Reduce semantic confusion matrix to rank 0 during DDP validation."""
        if RANK == -1 or not dist.is_available() or not dist.is_initialized():
            return
        if self.metrics.confusion_matrix.matrix is None:
            cm_nc = self.metrics.cm_nc
            self.metrics.confusion_matrix.matrix = torch.zeros((cm_nc, cm_nc), device=self.device, dtype=torch.int64)
        dist.reduce(self.metrics.confusion_matrix.matrix, dst=0, op=dist.ReduceOp.SUM)

    def save_pred_masks(self, preds: torch.Tensor, batch: dict[str, Any]) -> None:
        """Save semantic predictions as single-channel PNG masks."""
        if self.results_dir is None:
            return

        im_files = batch.get("im_file", [])
        if not im_files:
            return

        preds = preds.cpu().numpy()
        if isinstance(self.dataset, SemsegDataset) and self.dataset.label_mapping:
            preds = self.dataset.convert_label(preds, inverse=True)
        preds = preds.astype(np.uint8, copy=False)
        for pred, im_file in zip(preds, im_files):
            orig_shape = self.image_shapes.get(im_file)
            if orig_shape and pred.shape != orig_shape:
                pred = cv2.resize(pred, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)
            save_path = self.results_dir / Path(im_file).with_suffix(".png").name
            Image.fromarray(pred).save(save_path)

    def get_stats(self):
        """Return validation statistics.

        Returns:
            (dict): Dictionary of validation metrics.
        """
        return self.metrics.results_dict

    def get_desc(self):
        """Return a formatted description of evaluation metrics.

        Returns:
            (str): Formatted string with metric names.
        """
        return ("%22s" + "%11s" * 2) % ("Class", "mIoU", "PixAcc")

    def print_results(self):
        """Print validation results including per-class IoU."""
        pf = "%22s" + "%11.4f" * 2  # print format
        LOGGER.info(pf % ("all", self.metrics.miou, self.metrics.pixel_accuracy))
        # Per-class IoU
        if self.args.verbose and not self.training and self.nc > 1:
            per_class = self.metrics.per_class_iou
            for i, name in self.names.items():
                if i < len(per_class):
                    LOGGER.info(f"  {name}: IoU={per_class[i]:.4f}")
        if self.args.save_json and self.results_dir is not None:
            LOGGER.info(f"Semantic prediction masks saved to {self.results_dir}")

    def build_dataset(self, img_path, mode="val", batch=None):
        """Build semantic segmentation dataset.

        Args:
            img_path (str): Path to images.
            mode (str): Dataset mode.
            batch (int, optional): Batch size.

        Returns:
            (SemsegDataset): Dataset object.
        """
        use_rect = mode == "val"
        return SemsegDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            augment=False,
            hyp=self.args,
            cache=self.args.cache or None,
            data=self.data,
            rect=use_rect,
            batch_size=batch,
            stride=self.stride,
            pad=0,
            prefix=f"{mode}: ",
        )

    def get_dataloader(self, dataset_path, batch_size=16):
        """Build and return a dataloader for semantic segmentation validation.

        Args:
            dataset_path (str): Path to the dataset.
            batch_size (int): Batch size.

        Returns:
            (DataLoader): Validation dataloader.
        """
        dataset = self.build_dataset(dataset_path, batch=batch_size)
        return build_dataloader(dataset, batch=batch_size, workers=self.args.workers * 2, shuffle=False, rank=-1)
