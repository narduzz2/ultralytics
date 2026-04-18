# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import json
from collections import OrderedDict, namedtuple
from pathlib import Path

import numpy as np
import torch

from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements

from .base import BaseBackend


class TensorRTRTXBackend(BaseBackend):
    """NVIDIA TensorRT for RTX inference backend.

    Loads and runs inference with tensorrt_rtx serialized engines (.rtx.engine files). Engines are
    portable across RTX-class GPUs via runtime kernel compilation; a local kernel cache avoids
    paying the JIT cost on every load.
    """

    def load_model(self, weight: str | Path) -> None:
        """Load a TensorRT for RTX engine from a serialized .rtx.engine file."""
        LOGGER.info(f"Loading {weight} for TensorRT-RTX inference...")

        try:
            import tensorrt_rtx as trt
        except ImportError:
            check_requirements("tensorrt-rtx>=1.4.0")  # Linux/Windows x86_64 wheels only
            import tensorrt_rtx as trt

        if self.device.type == "cpu":
            self.device = torch.device("cuda:0")

        Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
        logger = trt.Logger(trt.Logger.INFO)

        with open(weight, "rb") as f, trt.Runtime(logger) as runtime:
            try:
                meta_len = int.from_bytes(f.read(4), byteorder="little")
                metadata = json.loads(f.read(meta_len).decode("utf-8"))
            except UnicodeDecodeError:
                f.seek(0)
                metadata = None
            engine = runtime.deserialize_cuda_engine(f.read())
            self.apply_metadata(metadata)

        if engine is None:
            raise RuntimeError(
                f"TensorRT-RTX failed to deserialize '{weight}' (tensorrt_rtx {trt.__version__}). "
                "See TRT log above for the underlying error. Common causes: tensorrt_rtx version mismatch "
                "between build host and target, unsupported target GPU architecture, or outdated NVIDIA driver."
            )
        self.context = engine.create_execution_context()

        self.bindings = OrderedDict()
        self.output_names = []
        self.fp16 = False
        self.dynamic = False

        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            shape = tuple(engine.get_tensor_shape(name))
            profile_shape = tuple(engine.get_tensor_profile_shape(name, 0)[2]) if is_input else None

            if is_input:
                if -1 in shape:
                    self.dynamic = True
                    self.context.set_input_shape(name, profile_shape)
                if dtype == np.float16:
                    self.fp16 = True
            else:
                self.output_names.append(name)

            shape = tuple(self.context.get_tensor_shape(name))
            im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(self.device)
            self.bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))

        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
        self.model = engine

    def forward(self, im: torch.Tensor) -> list[torch.Tensor]:
        """Run TensorRT-RTX inference with dynamic shape handling."""
        if self.dynamic and im.shape != self.bindings["images"].shape:
            self.context.set_input_shape("images", im.shape)
            self.bindings["images"] = self.bindings["images"]._replace(shape=im.shape)
            for name in self.output_names:
                self.bindings[name].data.resize_(tuple(self.context.get_tensor_shape(name)))

        s = self.bindings["images"].shape
        assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"

        self.binding_addrs["images"] = int(im.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        return [self.bindings[x].data for x in sorted(self.output_names)]
