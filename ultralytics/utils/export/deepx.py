# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from ultralytics.utils import LOGGER, YAML
from ultralytics.utils.checks import check_requirements


def onnx2deepx(
    onnx_file: str | Path,
    imgsz: tuple[int, int],
    dataset,
    metadata: dict | None = None,
    optimize: bool = False,
    prefix: str = "",
) -> Path:
    """Convert an ONNX model to DeepX format using the DeepX CLI tools.

    Args:
        onnx_file (str | Path): Input ONNX model path.
        imgsz (tuple[int, int]): Export image size as ``(height, width)``.
        dataset (DataLoader): Calibration dataloader used to build the DeepX config.
        metadata (dict | None, optional): Optional metadata to save as YAML. Defaults to None.
        optimize (bool, optional): If True, enables higher compiler optimization which reduces inference latency and
            increases compilation time. Defaults to False.
        prefix (str, optional): Logging prefix. Defaults to "".

    Returns:
        (Path): Path to the exported DeepX model directory.
    """
    try:
        import dx_com
    except ImportError:
        check_requirements("dx_com", cmds="-f https://sdk.deepx.ai/release/dxcom/v2.3.0/index.html")
        import dx_com
    import onnx

    LOGGER.info(f"\n{prefix} starting export with DeepX...")

    onnx_file = Path(onnx_file)
    export_path = Path(f"{onnx_file.stem}_deepx_model")
    export_path.mkdir(exist_ok=True)
    config_path = export_path / "config.json"

    input_name = onnx.load(str(onnx_file)).graph.input[0].name
    im_files = dataset.dataset.im_files
    with tempfile.TemporaryDirectory(prefix="deepx_calib_") as calib_dir:
        for i, src in enumerate(im_files):
            os.symlink(src, Path(calib_dir) / f"{i:08d}_{Path(src).name}")

        config = {
            "inputs": {input_name: [1, 3, imgsz[0], imgsz[1]]},
            "calibration_num": len(im_files),
            "calibration_method": "ema",
            "train_batchsize": 32,
            "num_samples": 1024,
            "default_loader": {
                "dataset_path": calib_dir,
                "file_extensions": ["jpeg", "jpg", "png", "JPEG"],
                "preprocessings": [
                    {"resize": {"mode": "pad", "size": imgsz[0], "pad_location": "edge", "pad_value": [114, 114, 114]}},
                    {"div": {"x": 255.0}},
                    {"convertColor": {"form": "BGR2RGB"}},
                    {"transpose": {"axis": [2, 0, 1]}},
                    {"expandDim": {"axis": 0}},
                ],
            },
        }

        with open(config_path, "w") as file:
            json.dump(config, file)

        dx_com.compile(
            model=str(onnx_file), output_dir=str(export_path), config=str(config_path), opt_level=int(optimize)
        )

    if metadata is not None:
        YAML.save(export_path / "metadata.yaml", metadata)

    return export_path
