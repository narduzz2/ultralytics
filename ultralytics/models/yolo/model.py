# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ultralytics.data.build import load_inference_source
from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.nn.modules.head import ADMBHead, AnomalyDetection
from ultralytics.nn.tasks import (
    ClassificationModel,
    DetectionModel,
    OBBModel,
    PoseModel,
    SegmentationModel,
    WorldModel,
    YOLOAnomalyModel,
    YOLOEModel,
    YOLOESegModel,
)
from ultralytics.utils import ROOT, YAML


class YOLO(Model):
    """YOLO (You Only Look Once) object detection model.

    This class provides a unified interface for YOLO models, automatically switching to specialized model types
    (YOLOWorld or YOLOE) based on the model filename. It supports various computer vision tasks including object
    detection, segmentation, classification, pose estimation, and oriented bounding box detection.

    Attributes:
        model: The loaded YOLO model instance.
        task: The task type (detect, segment, classify, pose, obb).
        overrides: Configuration overrides for the model.

    Methods:
        __init__: Initialize a YOLO model with automatic type detection.
        task_map: Map tasks to their corresponding model, trainer, validator, and predictor classes.

    Examples:
        Load a pretrained YOLO26n detection model
        >>> model = YOLO("yolo26n.pt")

        Load a pretrained YOLO26n segmentation model
        >>> model = YOLO("yolo26n-seg.pt")

        Initialize from a YAML configuration
        >>> model = YOLO("yolo26n.yaml")
    """

    def __init__(self, model: str | Path = "yolo26n.pt", task: str | None = None, verbose: bool = False):
        """Initialize a YOLO model.

        This constructor initializes a YOLO model, automatically switching to specialized model types (YOLOWorld or
        YOLOE) based on the model filename.

        Args:
            model (str | Path): Model name or path to model file, i.e. 'yolo26n.pt', 'yolo26n.yaml'.
            task (str, optional): YOLO task specification, i.e. 'detect', 'segment', 'classify', 'pose', 'obb'. Defaults
                to auto-detection based on model.
            verbose (bool): Display model info on load.
        """
        path = Path(model if isinstance(model, (str, Path)) else "")
        if "-world" in path.stem and path.suffix in {".pt", ".yaml", ".yml"}:  # if YOLOWorld PyTorch model
            new_instance = YOLOWorld(path, verbose=verbose)
            self.__class__ = type(new_instance)
            self.__dict__ = new_instance.__dict__
        elif "yoloe" in path.stem and path.suffix in {".pt", ".yaml", ".yml"}:  # if YOLOE PyTorch model
            new_instance = YOLOE(path, task=task, verbose=verbose)
            self.__class__ = type(new_instance)
            self.__dict__ = new_instance.__dict__
        else:
            # Continue with default YOLO initialization
            super().__init__(model=model, task=task, verbose=verbose)
            if hasattr(self.model, "model") and "RTDETR" in self.model.model[-1]._get_name():  # if RTDETR head
                from ultralytics import RTDETR

                new_instance = RTDETR(self)
                self.__class__ = type(new_instance)
                self.__dict__ = new_instance.__dict__

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "classify": {
                "model": ClassificationModel,
                "trainer": yolo.classify.ClassificationTrainer,
                "validator": yolo.classify.ClassificationValidator,
                "predictor": yolo.classify.ClassificationPredictor,
            },
            "detect": {
                "model": DetectionModel,
                "trainer": yolo.detect.DetectionTrainer,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            },
            "segment": {
                "model": SegmentationModel,
                "trainer": yolo.segment.SegmentationTrainer,
                "validator": yolo.segment.SegmentationValidator,
                "predictor": yolo.segment.SegmentationPredictor,
            },
            "pose": {
                "model": PoseModel,
                "trainer": yolo.pose.PoseTrainer,
                "validator": yolo.pose.PoseValidator,
                "predictor": yolo.pose.PosePredictor,
            },
            "obb": {
                "model": OBBModel,
                "trainer": yolo.obb.OBBTrainer,
                "validator": yolo.obb.OBBValidator,
                "predictor": yolo.obb.OBBPredictor,
            },
        }


class YOLOWorld(Model):
    """YOLO-World object detection model.

    YOLO-World is an open-vocabulary object detection model that can detect objects based on text descriptions without
    requiring training on specific classes. It extends the YOLO architecture to support real-time open-vocabulary
    detection.

    Attributes:
        model: The loaded YOLO-World model instance.
        task: Always set to 'detect' for object detection.
        overrides: Configuration overrides for the model.

    Methods:
        __init__: Initialize YOLOv8-World model with a pre-trained model file.
        task_map: Map tasks to their corresponding model, trainer, validator, and predictor classes.
        set_classes: Set the model's class names for detection.

    Examples:
        Load a YOLOv8-World model
        >>> model = YOLOWorld("yolov8s-world.pt")

        Set custom classes for detection
        >>> model.set_classes(["person", "car", "bicycle"])
    """

    def __init__(self, model: str | Path = "yolov8s-world.pt", verbose: bool = False) -> None:
        """Initialize YOLOv8-World model with a pre-trained model file.

        Loads a YOLOv8-World model for object detection. If no custom class names are provided, it assigns default COCO
        class names.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            verbose (bool): If True, prints additional information during initialization.
        """
        super().__init__(model=model, task="detect", verbose=verbose)

        # Assign default COCO class names when there are no custom names
        if not hasattr(self.model, "names"):
            self.model.names = YAML.load(ROOT / "cfg/datasets/coco8.yaml").get("names")

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "detect": {
                "model": WorldModel,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
                "trainer": yolo.world.WorldTrainer,
            }
        }

    def set_classes(self, classes: list[str]) -> None:
        """Set the model's class names for detection.

        Args:
            classes (list[str]): A list of categories i.e. ["person"].
        """
        self.model.set_classes(classes)
        # Remove background if it's given
        background = " "
        if background in classes:
            classes.remove(background)
        self.model.names = classes

        # Reset method class names
        if self.predictor:
            self.predictor.model.names = classes


class YOLOE(Model):
    """YOLOE object detection and segmentation model.

    YOLOE is an enhanced YOLO model that supports both object detection and instance segmentation tasks with improved
    performance and additional features like visual and text positional embeddings.

    Attributes:
        model: The loaded YOLOE model instance.
        task: The task type (detect or segment).
        overrides: Configuration overrides for the model.

    Methods:
        __init__: Initialize YOLOE model with a pre-trained model file.
        task_map: Map tasks to their corresponding model, trainer, validator, and predictor classes.
        get_text_pe: Get text positional embeddings for the given texts.
        get_visual_pe: Get visual positional embeddings for the given image and visual features.
        set_vocab: Set vocabulary and class names for the YOLOE model.
        get_vocab: Get vocabulary for the given class names.
        set_classes: Set the model's class names and embeddings for detection.
        val: Validate the model using text or visual prompts.
        predict: Run prediction on images, videos, directories, streams, etc.

    Examples:
        Load a YOLOE detection model
        >>> model = YOLOE("yoloe-11s-seg.pt")

        Set vocabulary and class names
        >>> model.set_vocab(["person", "car", "dog"], ["person", "car", "dog"])

        Predict with visual prompts
        >>> prompts = {"bboxes": [[10, 20, 100, 200]], "cls": ["person"]}
        >>> results = model.predict("image.jpg", visual_prompts=prompts)
    """

    def __init__(self, model: str | Path = "yoloe-11s-seg.pt", task: str | None = None, verbose: bool = False) -> None:
        """Initialize YOLOE model with a pre-trained model file.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            task (str, optional): Task type for the model. Auto-detected if None.
            verbose (bool): If True, prints additional information during initialization.
        """
        super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "detect": {
                "model": YOLOEModel,
                "validator": yolo.yoloe.YOLOEDetectValidator,
                "predictor": yolo.detect.DetectionPredictor,
                "trainer": yolo.yoloe.YOLOETrainer,
            },
            "segment": {
                "model": YOLOESegModel,
                "validator": yolo.yoloe.YOLOESegValidator,
                "predictor": yolo.segment.SegmentationPredictor,
                "trainer": yolo.yoloe.YOLOESegTrainer,
            },
        }

    def get_text_pe(self, texts):
        """Get text positional embeddings for the given texts."""
        assert isinstance(self.model, YOLOEModel)
        return self.model.get_text_pe(texts)

    def get_visual_pe(self, img, visual):
        """Get visual positional embeddings for the given image and visual features.

        This method extracts positional embeddings from visual features based on the input image. It requires that the
        model is an instance of YOLOEModel.

        Args:
            img (torch.Tensor): Input image tensor.
            visual (torch.Tensor): Visual features extracted from the image.

        Returns:
            (torch.Tensor): Visual positional embeddings.

        Examples:
            >>> model = YOLOE("yoloe-11s-seg.pt")
            >>> img = torch.rand(1, 3, 640, 640)
            >>> visual_features = torch.rand(1, 1, 80, 80)
            >>> pe = model.get_visual_pe(img, visual_features)
        """
        assert isinstance(self.model, YOLOEModel)
        return self.model.get_visual_pe(img, visual)

    def set_vocab(self, vocab: list[str], names: list[str]) -> None:
        """Set vocabulary and class names for the YOLOE model.

        This method configures the vocabulary and class names used by the model for text processing and classification
        tasks. The model must be an instance of YOLOEModel.

        Args:
            vocab (list[str]): Vocabulary list containing tokens or words used by the model for text processing.
            names (list[str]): List of class names that the model can detect or classify.

        Raises:
            AssertionError: If the model is not an instance of YOLOEModel.

        Examples:
            >>> model = YOLOE("yoloe-11s-seg.pt")
            >>> model.set_vocab(["person", "car", "dog"], ["person", "car", "dog"])
        """
        assert isinstance(self.model, YOLOEModel)
        self.model.set_vocab(vocab, names=names)

    def get_vocab(self, names):
        """Get vocabulary for the given class names."""
        assert isinstance(self.model, YOLOEModel)
        return self.model.get_vocab(names)

    def set_classes(self, classes: list[str], embeddings: torch.Tensor | None = None) -> None:
        """Set the model's class names and embeddings for detection.

        Args:
            classes (list[str]): A list of categories i.e. ["person"].
            embeddings (torch.Tensor, optional): Embeddings corresponding to the classes.
        """
        # Verify no background class is present
        assert " " not in classes
        assert isinstance(self.model, YOLOEModel)
        if sorted(list(self.model.names.values())) != sorted(classes):
            if embeddings is None:
                embeddings = self.get_text_pe(classes)  # generate text embeddings if not provided
            self.model.set_classes(classes, embeddings)

        # Reset method class names
        if self.predictor:
            self.predictor.model.names = self.model.names

    def val(
        self,
        validator=None,
        load_vp: bool = False,
        refer_data: str | None = None,
        **kwargs,
    ):
        """Validate the model using text or visual prompts.

        Args:
            validator (callable, optional): A callable validator function. If None, a default validator is loaded.
            load_vp (bool): Whether to load visual prompts. If False, text prompts are used.
            refer_data (str, optional): Path to the reference data for visual prompts.
            **kwargs (Any): Additional keyword arguments to override default settings.

        Returns:
            (dict): Validation statistics containing metrics computed during validation.
        """
        custom = {"rect": not load_vp}  # method defaults
        args = {**self.overrides, **custom, **kwargs, "mode": "val"}  # highest priority args on the right

        validator = (validator or self._smart_load("validator"))(args=args, _callbacks=self.callbacks)
        validator(model=self.model, load_vp=load_vp, refer_data=refer_data)
        self.metrics = validator.metrics
        return validator.metrics

    def predict(
        self,
        source=None,
        stream: bool = False,
        visual_prompts: dict[str, list] = {},
        refer_image=None,
        predictor=yolo.yoloe.YOLOEVPDetectPredictor,
        **kwargs,
    ):
        """Run prediction on images, videos, directories, streams, etc.

        Args:
            source (str | int | PIL.Image | np.ndarray, optional): Source for prediction. Accepts image paths, directory
                paths, URL/YouTube streams, PIL images, numpy arrays, or webcam indices.
            stream (bool): Whether to stream the prediction results. If True, results are yielded as a generator as they
                are computed.
            visual_prompts (dict[str, list]): Dictionary containing visual prompts for the model. Must include 'bboxes'
                and 'cls' keys when non-empty.
            refer_image (str | PIL.Image | np.ndarray, optional): Reference image for visual prompts.
            predictor (callable): Custom predictor class for visual prompt predictions. Defaults to
                YOLOEVPDetectPredictor.
            **kwargs (Any): Additional keyword arguments passed to the predictor.

        Returns:
            (list | generator): List of Results objects or generator of Results objects if stream=True.

        Examples:
            >>> model = YOLOE("yoloe-11s-seg.pt")
            >>> results = model.predict("path/to/image.jpg")
            >>> # With visual prompts
            >>> prompts = {"bboxes": [[10, 20, 100, 200]], "cls": ["person"]}
            >>> results = model.predict("path/to/image.jpg", visual_prompts=prompts)
        """
        if len(visual_prompts):
            assert "bboxes" in visual_prompts and "cls" in visual_prompts, (
                f"Expected 'bboxes' and 'cls' in visual prompts, but got {visual_prompts.keys()}"
            )
            assert len(visual_prompts["bboxes"]) == len(visual_prompts["cls"]), (
                f"Expected equal number of bounding boxes and classes, but got {len(visual_prompts['bboxes'])} and "
                f"{len(visual_prompts['cls'])} respectively"
            )
            if type(self.predictor) is not predictor:
                self.predictor = predictor(
                    overrides={
                        "task": self.model.task,
                        "mode": "predict",
                        "save": False,
                        "verbose": refer_image is None,
                        "batch": 1,
                        "device": kwargs.get("device", None),
                        "half": kwargs.get("half", False),
                        "imgsz": kwargs.get("imgsz", self.overrides.get("imgsz", 640)),
                    },
                    _callbacks=self.callbacks,
                )

            num_cls = (
                max(len(set(c)) for c in visual_prompts["cls"])
                if isinstance(source, list) and refer_image is None  # means multiple images
                else len(set(visual_prompts["cls"]))
            )
            self.model.model[-1].nc = num_cls
            self.model.names = [f"object{i}" for i in range(num_cls)]
            self.predictor.set_prompts(visual_prompts.copy())
            self.predictor.setup_model(model=self.model)

            if refer_image is None and source is not None:
                dataset = load_inference_source(source)
                if dataset.mode in {"video", "stream"}:
                    # NOTE: set the first frame as refer image for videos/streams inference
                    refer_image = next(iter(dataset))[1][0]
            if refer_image is not None:
                vpe = self.predictor.get_vpe(refer_image)
                self.model.set_classes(self.model.names, vpe)
                self.task = "segment" if isinstance(self.predictor, yolo.segment.SegmentationPredictor) else "detect"
                self.predictor = None  # reset predictor
        elif isinstance(self.predictor, yolo.yoloe.YOLOEVPDetectPredictor):
            self.predictor = None  # reset predictor if no visual prompts
        self.overrides["agnostic_nms"] = True  # use agnostic nms for YOLOE default

        return super().predict(source, stream, **kwargs)




class AnomalyValidator(yolo.detect.DetectionValidator):
    """Validator for YOLOAnomaly models.

    For end2end models (e.g. yoloe26n), non_max_suppression skips IoU-based NMS
    and only confidence-filters the top-k output, leaving overlapping boxes.
    This subclass applies real torchvision IoU NMS after confidence filtering.
    """

    def postprocess(self, preds):
        """Apply real IoU NMS for end2end models, then delegate."""
        if isinstance(preds, (tuple, list)):
            preds = preds[0]

        if self.end2end:
            import torchvision

            output = []
            for pred in preds:  # pred: [N, 6] = [x1, y1, x2, y2, score, cls]
                mask = pred[:, 4] > self.args.conf
                pred = pred[mask]
                if len(pred) == 0:
                    output.append({"bboxes": pred[:, :4], "conf": pred[:, 4], "cls": pred[:, 5], "extra": pred[:, 6:]})
                    continue
                keep = torchvision.ops.nms(pred[:, :4].float(), pred[:, 4].float(), self.args.iou)
                pred = pred[keep[: self.args.max_det]]
                output.append({"bboxes": pred[:, :4], "conf": pred[:, 4], "cls": pred[:, 5], "extra": pred[:, 6:]})
            return output

        return super().postprocess(preds)


class AnomalyPredictor(yolo.detect.DetectionPredictor):
    """Predictor for YOLOAnomaly models.

    Handles the (y, preds_dict) tuple that AnomalyDetection.forward() returns in
    non-export mode: extracts the tensor `y` before passing it to NMS / postprocess.
    """

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        """Unpack model output tuple then apply IoU NMS for end2end models."""
        # AnomalyDetection.forward() returns (y_tensor, preds_dict) in non-export mode.
        # y_tensor is already top-k selected by AnomalyDetection.postprocess (end2end path).
        if isinstance(preds, (tuple, list)):
            preds = preds[0]

        # For end2end models (e.g. yoloe26n), non_max_suppression skips IoU-based NMS and
        # only confidence-filters the top-k output, leaving overlapping boxes.  Apply real
        # NMS here so duplicates are removed before results are constructed.
        if getattr(self.model, "end2end", False):
            import torchvision

            output = []
            for pred in preds:  # pred: [N, 6] = [x1, y1, x2, y2, score, cls]
                mask = pred[:, 4] > self.args.conf
                pred = pred[mask]
                if len(pred) == 0:
                    output.append(pred)
                    continue
                keep = torchvision.ops.nms(pred[:, :4].float(), pred[:, 4].float(), self.args.iou)
                output.append(pred[keep[: self.args.max_det]])
            return self.construct_results(output, img, orig_imgs, **kwargs)

        return super().postprocess(preds, img, orig_imgs, **kwargs)


class YOLOAnomaly(Model):
    """
    YOLO-based training-free anomaly detection model.

    Loads any YOLOE-compatible pretrained model and converts it into anomaly detection
    mode using a memory bank of normal feature representations. No gradient-based
    training is needed: feed normal images via load_support_set() to populate the bank,
    then call predict().

    Attributes:
        model: The underlying YOLOAnomalyModel instance.

    Methods:
        __init__: Initialize from any YOLOE pretrained model file.
        task_map: Map tasks to model, validator, and predictor classes.
        setup: Configure anomaly detection with class names and threshold.
        load_support_set: Feed normal images to build the memory bank.
        save_mb: Save memory-bank payload to disk.
        load_mb: Load memory-bank payload from disk.
        reset_memory_bank: Clear the memory bank for reuse with a new support set.
        get_memory_bank_stats: Return memory bank statistics per detection head.

    Examples:
        One-shot anomaly detection workflow
        >>> model = YOLOAnomaly("yolo26s.pt")
        >>> model.setup(["defect"], conf=0.1)
        >>> model.load_support_set("datasets/mvtec/leather/train/good/")
        >>> results = model.predict("datasets/mvtec/leather/test/crack/")
    """

    def __init__(self, model: str | Path = "yoloe-11s.pt", verbose: bool = False) -> None:
        """
        Initialize YOLOAnomaly from a pretrained model file.

        Loads the checkpoint and automatically upgrades the underlying YOLOEModel to
        YOLOAnomalyModel to enable memory bank methods. If the loaded checkpoint was
        previously saved with anomaly metadata (via save()), the model is restored
        ready to predict without calling setup() or load_mb().

        Args:
            model (str | Path): Path to pretrained model (*.pt), e.g. 'yolo26s.pt'.
            verbose (bool): Print model info on load.

        Raises:
            AssertionError: If the loaded model is not a YOLOEModel instance.
        """
        super().__init__(model=model, task=None, verbose=verbose)
        if not isinstance(self.model, YOLOAnomalyModel):
            if isinstance(self.model, DetectionModel):
                self.model.__class__ = YOLOAnomalyModel
            else:
                raise AssertionError(
                    f"YOLOAnomaly requires a DetectionModel or YOLOEModel checkpoint, "
                    f"but loaded {type(self.model).__name__}."
                )

        # Restore anomaly metadata from checkpoint if previously saved via save()
        self._restore_anomaly_metadata()

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        """Map tasks to model, validator, and predictor classes."""
        return {
            "detect": {
                "model": DetectionModel,
                "predictor": AnomalyPredictor,
                "validator": AnomalyValidator,
            },
            # Segmentation checkpoints are supported as backbones; anomaly output is
            # always detection-shaped (boxes only), so we reuse AnomalyPredictor.
            "segment": {
                "model": SegmentationModel,
                "predictor": AnomalyPredictor,
                "validator": AnomalyValidator,
            },
        }

    def setup(self, names: list[str]) -> None:
        """
        Configure anomaly detection with class names.

        Must be called before load_support_set() and predict().

        Pass ``["anomaly"]`` (or any custom single name) to enable memory-bank cosine-similarity
        scoring (nc=1).  Pass ``["detect"]`` to use the model's original classification head
        scores with the original class names — useful for baseline comparison.

        Use set_ad_params() to configure the detection threshold (ad_conf) and max detections
        (ad_max_det) independently before running predict().

        Args:
            names (list[str]): Anomaly class names, e.g. ["anomaly"] or ["defect", "scratch"].
                Use ["detect"] as a special sentinel to start in original-classifier mode.
        """
        assert isinstance(self.model, YOLOAnomalyModel), (
            f"Expected YOLOAnomalyModel, got {type(self.model).__name__}. "
            "Ensure you loaded a YOLOE or plain YOLO detection model."
        )
        detect_mode = names == ["detect"]
        if detect_mode:
            # Use the model's current (original) class names so vocab embeddings are correct
            init_names = list(self.model.names.values())
        else:
            init_names = names
        self.model.setup_anomaly_detection(init_names)
        if detect_mode:
            self.model.set_anomaly_mode(False)  # confidence = original head scores
        # names are already set correctly inside setup_anomaly_detection / set_anomaly_mode

    def load_support_set(
        self,
        source,
        conf: float = 1e-6,
        imgsz: int = 640,
        device=None,
        verbose: bool = True,
        **kwargs,
    ) -> list[dict]:
        """
        Feed normal (non-anomalous) images to populate the memory bank.

        Memory bank is automatically frozen after this call. Run once before predict().

        Args:
            source: Image source - file path, directory, list of paths, etc.
            conf (float): Very low confidence to capture all candidate regions.
            imgsz (int): Inference image size.
            device: Device to run on (e.g. 'cuda:0', 'cpu').
            verbose (bool): Print memory bank stats after building.
            **kwargs: Additional keyword arguments passed to predict().

        Returns:
            list[dict]: Memory bank statistics per detection head.

        Examples:
            >>> model.load_support_set("datasets/mvtec/leather/train/good/")
        """
        from ultralytics.utils import LOGGER, TQDM

        def iter_support_sources(src):
            """Yield support images one-by-one so memory updates happen incrementally."""
            if isinstance(src, (list, tuple, set)):
                for item in src:
                    yield item
                return

            path = Path(src) if isinstance(src, (str, Path)) else None
            if path and path.is_dir():
                exts = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp", ".pfm"}
                for item in sorted(path.iterdir()):
                    if item.is_file() and item.suffix.lower() in exts:
                        yield str(item)
                return

            yield src

        assert isinstance(self.model, YOLOAnomalyModel), (
            "Call setup() before load_support_set()."
        )
        if verbose:
            LOGGER.info("YOLOAnomaly: building memory bank from support set...")
        self.model.set_memory_update(True)
        items = list(iter_support_sources(source))
        for item in TQDM(items, desc="Building memory bank"):
            self.predict(source=item, conf=conf, imgsz=imgsz, device=device, verbose=False, **kwargs)
        self.model.freeze_memory_bank()
        stats = self.model.get_memory_bank_stats()
        if verbose:
            for i, s in enumerate(stats):
                LOGGER.info(f"  Head[{i}]: {s['size']} features, dim={s['feature_dim']}")
        return stats

    def reset_memory_bank(self) -> None:
        """
        Clear the memory bank to allow rebuilding with a different support set.

        Does not require reloading the model.
        """
        assert isinstance(self.model, YOLOAnomalyModel)
        self.model.reset_memory_bank()

    def save_mb(self, path: str | Path) -> Path:
        """Save anomaly memory-bank payload for fast restore without rebuilding support set.

        Args:
            path (str | Path): Destination file path.

        Returns:
            Path: Saved file path.
        """
        assert isinstance(self.model, YOLOAnomalyModel), "Call setup() before save_mb()."
        heads_payload = []
        for h in self.model._get_ad_heads():
            heads_payload.append(
                {
                    "memory_bank": h.memory_bank.detach().cpu(),
                    "feature_dim": int(h.feature_dim) if h.feature_dim is not None else None,
                }
            )

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "admb_cache_v1",
            "heads": heads_payload,
            "num_heads": len(heads_payload),
            "names": dict(self.model.names) if getattr(self.model, "names", None) else None,
        }
        torch.save(payload, save_path)
        return save_path

    def load_mb(self, path: str | Path, freeze: bool = True, verbose: bool = True) -> list[dict]:
        """Load anomaly memory-bank payload previously saved by save_mb().

        Args:
            path (str | Path): Cache file path.
            freeze (bool): Freeze memory-bank updates after loading.
            verbose (bool): Print loaded stats.

        Returns:
            list[dict]: Per-head memory stats after loading.

        Raises:
            FileNotFoundError: If path does not exist.
            ValueError: If cache format is invalid or incompatible.
        """
        from ultralytics.utils import LOGGER

        assert isinstance(self.model, YOLOAnomalyModel), "Call setup() before load_mb()."
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Memory-bank cache not found: {load_path}")

        data = torch.load(load_path, map_location="cpu")
        if not isinstance(data, dict) or data.get("format") != "admb_cache_v1" or "heads" not in data:
            raise ValueError(f"Invalid memory-bank cache format in {load_path}")

        heads = self.model._get_ad_heads()
        if len(data["heads"]) != len(heads):
            raise ValueError(
                f"Memory-bank head count mismatch: cache={len(data['heads'])}, model={len(heads)}"
            )

        for h, hdata in zip(heads, data["heads"]):
            mb = hdata.get("memory_bank", None)
            if not isinstance(mb, torch.Tensor) or mb.dim() != 2:
                raise ValueError("Invalid memory_bank tensor in cache payload.")
            h.memory_bank = mb.to(h.memory_bank.device, dtype=h.memory_bank.dtype)
            fd = hdata.get("feature_dim", None)
            h.feature_dim = int(fd) if fd is not None else int(h.memory_bank.shape[1])

        if freeze:
            self.model.freeze_memory_bank()

        stats = self.model.get_memory_bank_stats()
        if verbose:
            LOGGER.info(f"YOLOAnomaly: loaded memory bank cache from {load_path}")
            for i, s in enumerate(stats):
                LOGGER.info(f"  Head[{i}]: {s['size']} features, dim={s['feature_dim']}")
        return stats

    def get_memory_bank_stats(self) -> list[dict]:
        """
        Return memory bank statistics for all detection heads.

        Returns:
            list[dict]: Per-head stats with keys 'size' and 'feature_dim'.
        """
        assert isinstance(self.model, YOLOAnomalyModel)
        return self.model.get_memory_bank_stats()

    def set_ad_params(
        self,
        ad_conf: float | None = None,
        ad_max_det: int | None = None,
        mode: str | None = None,
        accumulate_thresh: float | None = None,
        score_filter_kernel: int | None = None,
        active_layers: list[int] | None = None,
        auto_temperature: bool | None = None,
        calibration_interval: int | None = None,
        em_iters: int | None = None,
    ) -> None:
        """Set anomaly-detection inference parameters and optionally switch mode.

        Args:
            ad_conf (float | None): Confidence threshold for anomaly proposals.
            ad_max_det (int | None): Maximum number of detections per image.
            mode (str | None): If provided, switch to this mode ('anomaly' or 'detect').
            accumulate_thresh (float | None): Score threshold for OBMA memory-bank accumulation.
            score_filter_kernel (int | None): Kernel size for spatial mean filter (1=off, 3/5/7=smoothing).
            active_layers (list[int] | None): Indices of detection layers to enable, e.g. [0, 1] uses only
                                              the first two (P3+P4). None = all layers enabled.
            auto_temperature (bool | None): Enable/disable automatic temperature calibration from data.
                                            When True (default), β is derived from the 90th-percentile
                                            cosine similarity once the bank has enough features.
            calibration_interval (int | None): Re-calibrate every N support images (0 = calibrate once only).
            em_iters (int | None): EM iteration rounds during bank construction (1=single pass, default).
        """
        assert isinstance(self.model, YOLOAnomalyModel), "Call setup() before set_ad_params()."
        head = self.model.model[-1]
        assert isinstance(head, AnomalyDetection), "Call setup() before set_ad_params()."
        head.set_ad_params(
            ad_conf=ad_conf,
            ad_max_det=ad_max_det,
            accumulate_thresh=accumulate_thresh,
            score_filter_kernel=score_filter_kernel,
            active_layers=active_layers,
            auto_temperature=auto_temperature,
            calibration_interval=calibration_interval,
            em_iters=em_iters,
        )
        if mode is not None:
            self.set_mode(mode)

    def set_mode(self, mode: str) -> None:
        """
        Switch between anomaly detection and original classification mode.

        In 'anomaly' mode the model outputs a single anomaly score per region based
        on cosine distance to the memory bank (nc=1, ignores original class labels).
        In 'detect' mode the original classification head is restored so the model
        behaves as a standard detector — useful when the loaded weights already
        target specific defect classes.

        Call setup() before set_mode().

        Args:
            mode (str): 'anomaly' for memory-bank scoring, 'detect' for original classes.

        Examples:
            >>> model.set_mode("detect")   # use original defect class outputs
            >>> model.set_mode("anomaly")  # switch back to memory-bank scoring
        """
        assert mode in ("anomaly", "detect"), f"mode must be 'anomaly' or 'detect', got {mode!r}"
        assert isinstance(self.model, YOLOAnomalyModel), (
            "Call setup() before set_mode()."
        )
        self.model.set_anomaly_mode(mode == "anomaly")
        # Propagate updated names to the predictor's AutoBackend if already initialized,
        # so Results objects created on the next predict() use the correct class names.
        if self.predictor is not None and getattr(self.predictor, "model", None) is not None:
            self.predictor.model.names = self.model.names

    def predict(self, source=None, stream: bool = False, **kwargs):
        """
        Run anomaly detection on the given source.

        Detections are regions whose anomaly score exceeds the configured threshold.
        Ensure setup() and load_support_set() have been called beforehand,
        or load a previously saved anomaly model (.pt with embedded memory bank).

        Args:
            source: Image source for inference.
            stream (bool): Yield results as a generator instead of a list.
            **kwargs: Additional keyword arguments passed to the predictor.

        Returns:
            list[Results] | generator: Anomaly detection results.
        """
        return super().predict(source=source, stream=stream, **kwargs)

    @property
    def is_configured(self) -> bool:
        """Check if the model is fully configured for anomaly detection (head + memory bank populated)."""
        head = self.model.model[-1]
        if not isinstance(head, AnomalyDetection) or head.adhead is None:
            return False
        # Check at least one head has real features (beyond the 10-entry padding from _memory_tensor)
        for h in head.adhead:
            if isinstance(h, ADMBHead) and h.memory_bank.numel() > 0 and h.memory_bank.shape[0] > 10:
                return True
        return False

    def save(self, filename: str | Path = "saved_model.pt") -> None:
        """Save the anomaly model with memory bank and metadata embedded in the checkpoint.

        The saved .pt file can be loaded directly with ``YOLOAnomaly(path)`` and used
        for prediction without calling ``setup()`` or ``load_mb()``.

        Args:
            filename (str | Path): Destination file path.
        """
        from copy import deepcopy
        from datetime import datetime
        from ultralytics import __version__

        self._check_is_pytorch_model()

        # Build anomaly metadata to persist alongside the model
        head = self.model.model[-1]
        anomaly_meta = {}
        if isinstance(head, AnomalyDetection) and head.adhead is not None:
            anomaly_meta = {
                "anomaly_names": getattr(self.model, "_anomaly_names", dict(self.model.names)),
                "original_names": getattr(self.model, "_original_names", {}),
                "original_nc": getattr(self.model, "_original_nc", getattr(head, "original_nc", head.nc)),
                "anomaly_mode": head.anomaly_mode,
                "ad_conf": head.ad_conf,
                "ad_max_det": head.ad_max_det,
                "temperature": head.temperature,
                "K": head.K,
                "accumulate_thresh": head.accumulate_thresh,
                "score_filter_kernel": head.score_filter_kernel,
            }

        updates = {
            "model": deepcopy(self.model).half() if isinstance(self.model, torch.nn.Module) else self.model,
            "date": datetime.now().isoformat(),
            "version": __version__,
            "license": "AGPL-3.0 License (https://ultralytics.com/license)",
            "docs": "https://docs.ultralytics.com",
            "anomaly_meta": anomaly_meta,
        }
        torch.save({**self.ckpt, **updates}, filename)

    def _restore_anomaly_metadata(self) -> None:
        """Restore anomaly detection state from checkpoint metadata if available.

        Called automatically by __init__ when loading a previously saved anomaly model.
        If the checkpoint contains anomaly_meta and the model head is already an
        AnomalyDetection with populated adhead, the model is ready to predict immediately.
        """
        from ultralytics.utils import LOGGER

        ckpt = self.ckpt
        if not ckpt or not isinstance(ckpt, dict):
            return

        meta = ckpt.get("anomaly_meta")
        head = self.model.model[-1]
        is_ad_head = isinstance(head, AnomalyDetection) and head.adhead is not None

        if not is_ad_head and meta is None:
            return  # Fresh model, no anomaly state to restore

        if is_ad_head:
            # Model was saved with AnomalyDetection head — restore metadata
            if meta:
                self.model._anomaly_names = meta.get("anomaly_names", {0: "anomaly"})
                self.model._original_names = meta.get("original_names", {})
                self.model._original_nc = meta.get("original_nc", head.nc)
                head.original_nc = self.model._original_nc
                head.ad_conf = meta.get("ad_conf", 0.5)
                head.ad_max_det = meta.get("ad_max_det", 9)
                head.temperature = meta.get("temperature", 3.0)
                head.K = meta.get("K", 15)
                head.accumulate_thresh = meta.get("accumulate_thresh", 0.4)
                head.score_filter_kernel = meta.get("score_filter_kernel", 1)

                # Propagate params to ADMBHead sub-modules
                for h in head.adhead:
                    if isinstance(h, ADMBHead):
                        h.temperature = head.temperature
                        h.K = head.K
                        h.accumulate_thresh = head.accumulate_thresh
                        h.score_filter_kernel = head.score_filter_kernel

                anomaly_mode = meta.get("anomaly_mode", True)
                head.set_anomaly_mode(anomaly_mode)
                if anomaly_mode:
                    self.model.names = {0: list(self.model._anomaly_names.values())[0]}
                else:
                    self.model.names = self.model._original_names

            # Freeze memory bank (loaded model should not accumulate)
            self.model.freeze_memory_bank()

            has_mb = any(
                h.memory_bank.numel() > 0 and h.memory_bank.shape[0] > 10
                for h in head.adhead if isinstance(h, ADMBHead)
            )
            if has_mb:
                stats = self.model.get_memory_bank_stats()
                LOGGER.info("YOLOAnomaly: restored anomaly model from checkpoint")
                for i, s in enumerate(stats):
                    LOGGER.info(f"  Head[{i}]: {s['size']} features, dim={s['feature_dim']}")


