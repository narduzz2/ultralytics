---
comments: true
description: Learn how to export Ultralytics YOLO models to DeepX format for efficient deployment on DeepX NPU hardware with INT8 quantization and high-performance edge inference.
keywords: DeepX, NPU, model export, Ultralytics, YOLO, edge AI, INT8 quantization, embedded inference, dx_com, dx_engine, dxnn, deep learning, hardware acceleration
---

# DeepX Export for Ultralytics YOLO Models

Deploying computer vision models on specialized NPU hardware requires a compatible and optimized model format. Exporting [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) models to DeepX format enables efficient, INT8-quantized inference on DeepX NPU accelerators. This guide walks you through converting your YOLO models to DeepX format and deploying them on DeepX-powered hardware.

## What is DeepX?

[DeepX](https://www.deepx.ai/) is an AI semiconductor company specializing in Neural Processing Units (NPUs) designed for power-efficient deep learning inference at the edge. DeepX NPUs are engineered for demanding embedded and industrial AI applications, delivering high throughput with minimal power consumption. Their hardware is well suited for deployment scenarios where cloud connectivity is unreliable or undesirable, such as robotics, smart cameras, and industrial automation systems.

## DeepX Export Format

The DeepX export produces a compiled `.dxnn` model binary that is optimized for execution on DeepX NPU hardware. The compilation pipeline uses the `dx_com` toolkit to perform INT8 quantization and hardware-specific optimization, generating a self-contained model directory ready for deployment.

## Key Features of DeepX Models

DeepX models offer several advantages for edge deployment:

- **INT8 Quantization**: Models are quantized to INT8 precision during export, significantly reducing model size and maximizing NPU throughput.
- **NPU-Optimized**: The `.dxnn` format is specifically compiled for DeepX NPU hardware, leveraging dedicated acceleration units for fast, efficient inference.
- **Low Power Consumption**: By offloading inference to the NPU, DeepX models consume far less power than equivalent CPU or GPU inference.
- **Calibration-Based Accuracy**: The export uses EMA-based calibration with real dataset images to minimize accuracy loss during quantization.
- **Self-Contained Output**: The exported model directory bundles the compiled binary, calibration config, and metadata for straightforward deployment.

## Export to DeepX: Converting Your YOLO Model

Export an Ultralytics YOLO model to DeepX format and run inference with the exported model.

!!! note

    DeepX export is only supported on x86-64 Linux machines. ARM64 (aarch64) is not supported for the export step.

### Installation

To install the required packages, run:

!!! tip "Installation"

    === "CLI"

        ```bash
        # Install the required package for YOLO
        pip install ultralytics
        ```

The `dx_com` compiler package will be automatically installed from the DeepX SDK repository on first export. For detailed instructions and best practices related to the installation process, check our [Ultralytics Installation guide](../quickstart.md). While installing the required packages for YOLO, if you encounter any difficulties, consult our [Common Issues guide](../guides/yolo-common-issues.md) for solutions and tips.

### Usage

!!! example "Usage"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the YOLO26 model
        model = YOLO("yolo26n.pt")

        # Export the model to DeepX format (int8=True is enforced automatically)
        model.export(format="deepx")  # creates 'yolo26n_deepx_model/'
        ```

    === "CLI"

        ```bash
        # Export a YOLO26n PyTorch model to DeepX format
        yolo export model=yolo26n.pt format=deepx # creates 'yolo26n_deepx_model/'
        ```

### Export Arguments

| Argument   | Type             | Default          | Description                                                                                                                                                                                                                                                      |
| :--------- | :--------------- | :--------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format`   | `str`            | `'deepx'`        | Target format for the exported model, defining compatibility with DeepX NPU hardware.                                                                                                                                                                            |
| `imgsz`    | `int` or `tuple` | `640`            | Desired image size for the model input. DeepX export requires a square input — pass an integer (e.g., `640`) or a tuple where height equals width.                                                                                                               |
| `batch`    | `int`            | `1`              | Specifies export model batch inference size or the max number of images the exported model will process concurrently in `predict` mode.                                                                                                                          |
| `int8`     | `bool`           | `True`           | Enables INT8 quantization. Required for DeepX export — automatically set to `True` if not specified.                                                                                                                                                             |
| `data`     | `str`            | `'coco128.yaml'` | Dataset configuration file used for INT8 calibration. Specifies the calibration image source.                                                                                                                                                                    |
| `fraction` | `float`          | `1.0`            | Specifies the fraction of the dataset to use for INT8 quantization calibration. Allows for calibrating on a subset of the full dataset, useful for experiments or when resources are limited. If not specified with INT8 enabled, the full dataset will be used. |
| `device`   | `str`            | `None`           | Specifies the device for exporting: GPU (`device=0`) or CPU (`device=cpu`).                                                                                                                                                                                      |
| `optimize` | `bool`           | `False`          | Enables higher compiler optimization which reduces inference latency and increases compilation time.                                                                                                                                                             |

!!! tip

    Always run DeepX export on an **x86-64 Linux** host. The `dx_com` compiler does not support ARM64.

For more details about the export process, visit the [Ultralytics documentation page on exporting](../modes/export.md).

### Output Structure

After a successful export, a model directory is created with the following layout:

```
yolo26n_deepx_model/
├── yolo26n.dxnn     # Compiled DeepX model binary (NPU executable)
├── config.json      # Calibration and preprocessing configuration
└── metadata.yaml    # Model metadata (classes, image size, task, etc.)
```

The `.dxnn` file is the compiled model binary that the `dx_engine` runtime loads directly on the NPU. The `metadata.yaml` contains class names, image size, and other information used by the Ultralytics inference pipeline.

## Deploying Exported YOLO DeepX Models

Once you've successfully exported your Ultralytics YOLO model to DeepX format, the next step is deploying these models on DeepX NPU hardware.

### Runtime Installation

Inference requires the DeepX NPU driver, the `libdxrt` runtime, and the `dx_engine` Python package.

!!! note

    DeepX runtime is only supported on x86-64 Linux machines and ARM64 Debian Trixie machines (Raspberry Pi 5).

```bash
# Build dependencies for the DKMS NPU driver module
sudo apt update
sudo apt install -y dkms

# Install the NPU driver and libdxrt runtime
wget https://github.com/DEEPX-AI/dx_rt_npu_linux_driver/raw/main/release/2.4.0/dxrt-driver-dkms_2.4.0-2_all.deb
sudo dpkg -i dxrt-driver-dkms_2.4.0-2_all.deb
wget https://github.com/DEEPX-AI/dx_rt/raw/main/release/3.3.1/libdxrt_3.3.1_all.deb
sudo dpkg -i libdxrt_3.3.1_all.deb

# Install the bundled dx_engine Python wheel
pip install /usr/share/libdxrt/src/python_package/dx_engine-*.whl
```

Verify the runtime is installed correctly with `dxrt-cli --version`. You should see output similar to:

```sh
DXRT v3.3.1
Minimum Driver Versions
Device Driver: v2.4.0
PCIe Driver: v2.2.0
Firmware: v2.5.2
Minimum Compiler Versions
Compiler: v1.18.1
.dxnn File Format: v6
```

### Usage

!!! example "Usage"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported DeepX model
        model = YOLO("yolo26n_deepx_model")

        # Run inference
        results = model("https://ultralytics.com/images/bus.jpg")

        # Process results
        for r in results:
            print(f"Detected {len(r.boxes)} objects")
            r.show()
        ```

    === "CLI"

        ```bash
        # Run inference with the exported DeepX model
        yolo predict model='yolo26n_deepx_model' source='https://ultralytics.com/images/bus.jpg'
        ```

### Visualizing with dxtron

[dxtron](https://sdk.deepx.ai/) is DeepX's graph visualizer for inspecting the compiled `.dxnn` model.

Install `dxtron` on x86-64 Linux by downloading the `.deb` package from the DeepX SDK and installing it via `dpkg`:

```bash
wget https://sdk.deepx.ai/release/dxtron/v2.0.1/dxtron_2.0.1_amd64.deb
sudo dpkg -i dxtron_2.0.1_amd64.deb
```

Then open your exported model:

```bash
dxtron yolo26n_deepx_model/yolo26n.dxnn
```

!!! note

    `dxtron` is only available for **x86-64 Linux**. ARM64/aarch64 and non-Linux platforms are not supported.

## Real-World Applications

YOLO models deployed on DeepX NPU hardware are well suited for a wide range of edge AI applications:

- **Smart Surveillance**: Real-time object detection for security and monitoring systems with low power consumption and no cloud dependency.
- **Industrial Automation**: On-device quality control, defect detection, and process monitoring in factory environments.
- **Robotics**: Vision-based navigation, obstacle avoidance, and object recognition on autonomous robots and drones.
- **Smart Agriculture**: Crop health monitoring, pest detection, and yield estimation using [computer vision in agriculture](https://www.ultralytics.com/solutions/ai-in-agriculture).
- **Retail Analytics**: Customer flow analysis, shelf monitoring, and inventory tracking with real-time edge inference.

## Summary

In this guide, you've learned how to export Ultralytics YOLO models to DeepX format and deploy them on DeepX NPU hardware. The export pipeline uses INT8 calibration and the `dx_com` compiler to produce a hardware-optimized `.dxnn` binary, while the `dx_engine` runtime handles inference on the device.

The combination of [Ultralytics YOLO](https://www.ultralytics.com/yolo) and DeepX's NPU technology provides an effective solution for running advanced computer vision workloads on embedded and edge devices — delivering high throughput with low power consumption for real-time applications.

For further details on usage, visit the [DeepX official website](https://www.deepx.ai/).

Also, if you'd like to know more about other Ultralytics YOLO integrations, visit our [integration guide page](../integrations/index.md). You'll find plenty of useful resources and insights there.

## FAQ

### How do I export my Ultralytics YOLO model to DeepX format?

You can export your model using the `export()` method in Python or via the CLI. The export automatically enables INT8 quantization and uses a calibration dataset to minimize accuracy loss. The `dx_com` compiler package is installed automatically if not already present.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")
        model.export(format="deepx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n.pt format=deepx
        ```

### Why does DeepX export require INT8 quantization?

DeepX NPUs are designed to execute INT8 computations at maximum efficiency. The `dx_com` compiler quantizes the model during export using EMA-based calibration with real dataset images, enabling the NPU to deliver its full performance. INT8 is always enforced for DeepX exports — if you pass `int8=False`, it will be overridden with a warning.

### What platforms are supported for DeepX export?

DeepX model export (compilation) requires an **x86-64 Linux** host. The export step is not supported on ARM64, aarch64 and Windows machines. Inference using the exported `.dxnn` model can be run on any Linux platform (x86-64 and ARM64) supported by the `dx_engine` runtime.

### What is the output of a DeepX export?

The export creates a directory (e.g., `yolo26n_deepx_model/`) containing:

- `yolo26n.dxnn` — the compiled NPU binary
- `config.json` — calibration and preprocessing settings
- `metadata.yaml` — model metadata including class names and image size

### Can I deploy custom-trained models on DeepX hardware?

Yes. Any model trained using [Ultralytics Train Mode](../modes/train.md) and exported with `format="deepx"` can be deployed on DeepX NPU hardware, provided it uses supported layer operations. Export supports detection, segmentation, pose estimation, oriented bounding box (OBB), and classification tasks.

### How many calibration images should I use for DeepX export?

The DeepX export pipeline uses every image in the calibration dataset (after `fraction` filtering) with the EMA calibration method. A few hundred images is usually sufficient for good quantization accuracy. Point `data` at a smaller dataset (or set `fraction` below `1.0`) if compilation time becomes a concern on large datasets.

### How do I install the DeepX runtime for inference?

The DeepX runtime is not bundled with `ultralytics` and must be installed separately before running inference. On x86-64 Linux machines and ARM64 Debian Trixie machines (Raspberry Pi 5), install the NPU driver (`dxrt-driver-dkms`) and runtime (`libdxrt`) from the DEEPX-AI GitHub releases, then install the bundled `dx_engine` Python wheel. See the [Runtime Installation](#runtime-installation) section above for step-by-step commands.
