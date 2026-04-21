# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import re

from ultralytics.models.yolo.classify.predict import ClassificationPredictor
from ultralytics.utils import DEFAULT_CFG


class ReidPredictor(ClassificationPredictor):
    """Predictor for person re-identification models.

    Inherits image preprocessing, source setup, and Results construction from
    ClassificationPredictor. The only ReID-specific override is ``write_results``,
    which displays the embedding dimensionality rather than a misleading top-k
    classification string. Model outputs may be a tensor or an ``(embedding,
    feat_bn)`` tuple; the classification postprocess already handles both by
    taking ``preds[0]`` when a tuple is returned.

    Examples:
        >>> from ultralytics.models.yolo.reid import ReidPredictor
        >>> args = dict(model="yolo26n-reid.pt", source="path/to/query/")
        >>> predictor = ReidPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize ReidPredictor and set task to 'reid'."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "reid"

    def write_results(self, i, p, im, s):
        """Write results with an embedding summary instead of class probabilities."""
        string = ""
        if len(im.shape) == 3:
            im = im[None]
        if self.source_type.stream or self.source_type.from_img or self.source_type.tensor:
            string += f"{i}: "
            frame = self.dataset.count
        else:
            match = re.search(r"frame (\d+)/", s[i])
            frame = int(match[1]) if match else None

        self.txt_path = self.save_dir / "labels" / (p.stem + ("" if self.dataset.mode == "image" else f"_{frame}"))
        string += "{:g}x{:g} ".format(*im.shape[2:])
        result = self.results[i]
        result.save_dir = self.save_dir.__str__()

        emb_dim = result.probs.data.shape[0] if result.probs is not None else 0
        string += f"embedding({emb_dim}-d), {result.speed['inference']:.1f}ms"

        if self.args.save or self.args.show:
            self.plotted_img = result.plot(line_width=self.args.line_width, probs=False)
        if self.args.save_txt:
            result.save_txt(f"{self.txt_path}.txt", save_conf=self.args.save_conf)
        if self.args.save_crop:
            result.save_crop(save_dir=self.save_dir / "crops", file_name=self.txt_path.stem)
        if self.args.show:
            self.show(str(p))
        if self.args.save:
            self.save_predicted_images(self.save_dir / p.name, frame)
        return string
