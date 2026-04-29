# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math

import cv2
import numpy as np

from ultralytics.data.augment import LetterBox
from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils import ops


class SemanticSegmentationPredictor(BasePredictor):
    """Predictor for semantic segmentation models.

    This predictor processes model outputs to produce per-pixel class label maps.

    Examples:
        >>> from ultralytics.models.yolo.semseg import SemanticSegmentationPredictor
        >>> args = dict(model="yolo26n-semseg.pt", source="path/to/image.jpg")
        >>> predictor = SemanticSegmentationPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize SemanticSegmentationPredictor.

        Args:
            cfg (dict): Configuration for the predictor.
            overrides (dict, optional): Configuration overrides.
            _callbacks (list, optional): Callback functions.
        """
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "semseg"

    def pre_transform(self, im: list[np.ndarray]) -> list[np.ndarray]:
        """Short-side scale to imgsz and pad to stride multiples."""
        imgsz = self.imgsz[0] if isinstance(self.imgsz, (list, tuple)) else self.imgsz
        stride_t = self.model.stride
        stride = int(stride_t.max() if hasattr(stride_t, "max") else stride_t)

        # Static-shape backend (e.g. OpenVINO/TensorRT exported with dynamic=False):
        # model input is fixed to (imgsz, imgsz); fall back to square letterbox.
        if getattr(self.model, "dynamic", True) is False:
            letterbox = LetterBox(new_shape=(imgsz, imgsz), auto=False, scaleup=False, stride=stride)
            return [letterbox(image=x) for x in im]

        scaled = []
        for x in im:
            h0, w0 = x.shape[:2]
            r = imgsz / min(h0, w0)
            if r != 1:
                if h0 < w0:
                    h, w = imgsz, math.ceil(w0 * r)
                else:
                    h, w = math.ceil(h0 * r), imgsz
                x = cv2.resize(x, (w, h), interpolation=cv2.INTER_LINEAR)
            scaled.append(x)

        rect_h = math.ceil(max(x.shape[0] for x in scaled) / stride) * stride
        rect_w = math.ceil(max(x.shape[1] for x in scaled) / stride) * stride
        letterbox = LetterBox(auto=False, scaleup=False, center=False, stride=stride)
        return [letterbox({"rect_shape": (rect_h, rect_w)}, image=x) for x in scaled]

    def postprocess(self, preds, img, orig_imgs):
        """Convert model logits to semantic segmentation results.

        Args:
            preds (torch.Tensor | tuple): Model output logits.
            img (torch.Tensor): Preprocessed input image tensor.
            orig_imgs (list | torch.Tensor): Original images.

        Returns:
            (list[Results]): List of Results objects with semantic masks.
        """
        if isinstance(preds, (tuple, list)):
            preds = preds[0]

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

        results = []
        for i, (pred, orig_img) in enumerate(zip(preds, orig_imgs)):
            img_path = self.batch[0][i] if isinstance(self.batch[0], list) else self.batch[0]
            # pred: [nc, H, W] logits on letterboxed input. Remove padding, then resize to original image.
            oh, ow = orig_img.shape[:2]
            pred = ops.scale_masks(pred.unsqueeze(0), (oh, ow))[0]
            class_map = pred.argmax(0).cpu()  # [H, W]
            results.append(Results(orig_img, path=img_path, names=self.model.names, semantic_mask=class_map))
        return results
