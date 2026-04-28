# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from ultralytics.utils import ARM64, IS_DEBIAN_TRIXIE, LOGGER
from ultralytics.utils.checks import check_apt_requirements, install_deb, is_sudo_available

from .base import BaseBackend

# DeepX runtime CLI and install locations
DXRT_CMD = ["dxrt-cli", "--version"]
SIXFAB_WHEEL_DIR = Path("/opt/sixfab-dx/wheels")  # arm64 Trixie wheel location (from sixfab-dx APT package)
LIBDXRT_WHEEL_DIR = Path("/usr/share/libdxrt/src/python_package")  # x86-64 wheel location (from libdxrt .deb)

# Download URLs
SIXFAB_REPO_URL = "https://github.com/sixfab/sixfab_dx/"
DRIVER_DEB_URL = (
    "https://github.com/DEEPX-AI/dx_rt_npu_linux_driver/raw/main/release/2.4.0/dxrt-driver-dkms_2.4.0-2_all.deb"
)
RUNTIME_DEB_URL = "https://github.com/DEEPX-AI/dx_rt/raw/main/release/3.3.0/libdxrt_3.3.0_all.deb"
DEEPX_DOCS_URL = "https://docs.ultralytics.com/integrations/deepx/"


class DeepXBackend(BaseBackend):
    """DeepX NPU inference backend for DeepX hardware accelerators.

    Loads compiled DeepX models (.dxnn files) and runs inference using the DeepX dx_engine runtime. Requires the
    dx_engine package to be installed.
    """

    def load_model(self, weight: str | Path) -> None:
        """Load a DeepX model from a directory containing a .dxnn file.

        Auto-installs the DeepX runtime and ``dx_engine`` Python package on supported platforms:
        arm64 Debian Trixie via the Sixfab APT repository, or x86-64 Linux via the NPU driver and libdxrt
        ``.deb`` packages from the DEEPX-AI GitHub releases.

        Args:
            weight (str | Path): Path to the DeepX model directory containing the .dxnn binary.

        Raises:
            FileNotFoundError: If no .dxnn file is found in the given directory.
            OSError: If the ``dx_engine`` wheel cannot be located after runtime install.
        """
        # 1. DeepX runtime (dxrt-cli): install if missing
        try:
            subprocess.run(DXRT_CMD, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            if IS_DEBIAN_TRIXIE and ARM64:
                # arm64 Trixie: install sixfab-dx via APT
                LOGGER.info(f"\nDeepX inference requires DeepX runtime. Attempting install from {SIXFAB_REPO_URL}")
                sudo = "sudo " if is_sudo_available() else ""
                for c in (
                    f"wget -qO - https://sixfab.github.io/sixfab_dx/public.gpg | {sudo}gpg --dearmor -o /usr/share/keyrings/sixfab-dx.gpg",
                    f'echo "deb [signed-by=/usr/share/keyrings/sixfab-dx.gpg] https://sixfab.github.io/sixfab_dx trixie main" | {sudo}tee /etc/apt/sources.list.d/sixfab-dx.list',
                ):
                    subprocess.run(c, shell=True, check=True, stdout=subprocess.DEVNULL)
                check_apt_requirements(["sixfab-dx"])
            else:
                # x86-64 Linux: download and dpkg install NPU driver + libdxrt runtime from GitHub
                LOGGER.info("DeepX runtime not found. Attempting to install NPU driver and libdxrt...")
                with tempfile.TemporaryDirectory() as tmpdir:
                    install_deb(DRIVER_DEB_URL, Path(tmpdir), "NPU driver")
                    install_deb(RUNTIME_DEB_URL, Path(tmpdir), "runtime (libdxrt)")

        # 2. dx_engine Python package: pip install bundled wheel if missing
        try:
            from dx_engine import InferenceEngine
        except ImportError:
            wheel_dir = SIXFAB_WHEEL_DIR if (IS_DEBIAN_TRIXIE and ARM64) else LIBDXRT_WHEEL_DIR
            wheels = sorted(wheel_dir.glob("dx_engine-*.whl")) if wheel_dir.exists() else []
            if not wheels:
                raise OSError(
                    f"dx_engine wheel not found in {wheel_dir}. Runtime auto-install may have failed. "
                    f"For manual setup, see {DEEPX_DOCS_URL}"
                )
            LOGGER.info(f"DeepX inference requires dx_engine. Attempting to install from {wheels[-1]}")
            subprocess.run([sys.executable, "-m", "pip", "install", str(wheels[-1])], check=True)
            from dx_engine import InferenceEngine

        # Log runtime version if available
        try:
            out = subprocess.run(DXRT_CMD, capture_output=True, check=True).stdout.decode()
            suffix = f" (runtime v{out.splitlines()[0].split()[-1].lstrip('v')})"
        except (FileNotFoundError, subprocess.CalledProcessError, IndexError):
            suffix = ""
        LOGGER.info(f"Loading {weight} for DeepX inference...{suffix}")

        w = Path(weight)
        found = next(w.rglob("*.dxnn"), None)
        if found is None:
            raise FileNotFoundError(f"No .dxnn file found in: {w}")

        self.model = InferenceEngine(str(found))

        # Load metadata
        metadata_file = found.parent / "metadata.yaml"
        if metadata_file.exists():
            from ultralytics.utils import YAML

            self.apply_metadata(YAML.load(metadata_file))

    def forward(self, im: torch.Tensor) -> np.ndarray | list[np.ndarray]:
        """Run inference on the DeepX NPU.

        Converts each image from BCHW float [0, 1] to HWC uint8 [0, 255] per the DeepX runtime contract,
        runs the engine per image, then stacks outputs along the batch dimension.

        Args:
            im (torch.Tensor): Input image tensor in BCHW format, normalized to [0, 1].

        Returns:
            (np.ndarray | list[np.ndarray]): Model predictions as a single array or list of arrays.
        """
        outputs = []
        for sample in im.cpu().numpy():
            sample = np.ascontiguousarray(np.clip(np.transpose(sample, (1, 2, 0)) * 255, 0, 255).astype(np.uint8))
            for i, out in enumerate(map(np.asarray, self.model.run([sample]))):
                if i == len(outputs):
                    outputs.append([])
                outputs[i].append(out if out.ndim and out.shape[0] == 1 else out[None])
        y = [np.concatenate(x, axis=0) for x in outputs]
        return y[0] if len(y) == 1 else y
