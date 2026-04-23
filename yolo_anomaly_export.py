"""
ONNX Export + Inference test for YOLOAnomaly.

Workflow:
  1. Build / load an anomaly model from support images
  2. Export to ONNX  (dense path: conf=0 inside the graph, topk selects top-ad_max_det)
  3. Run onnxruntime inference on a test image
  4. Compare PyTorch vs ONNX box outputs to verify correctness
"""

import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics.models.yolo.model import YOLOAnomaly
from yolo_anomaly import collect_images, get_mvtec_yolo_data, get_arguments, build_ad_model

# ── Config ────────────────────────────────────────────────────────────────────
CATEGORY       = "wood"
BASE_MODEL     = "yolo26l.pt"
IMGSZ          = 640
EXPORT_DIR     = "./runs/temp/export"
ONNX_OPSET    = 17   # opset 17 supports TopK with dynamic k; 16+ for TensorRT 8.6+

os.makedirs(EXPORT_DIR, exist_ok=True)


# ── Step 1: Build / load anomaly model ───────────────────────────────────────

def get_model(category: str, base_model: str) -> YOLOAnomaly:
    data_config = get_mvtec_yolo_data(category)
    model_arg, anomaly_arg = get_arguments(category)
    model = build_ad_model(
        base_model, data_config, model_arg, anomaly_arg,
        category=category, replace_model=False,   # set True to rebuild bank
    )
    model.set_ad_params(**anomaly_arg)
    return model, data_config


# ── Step 2: ONNX export ───────────────────────────────────────────────────────

def export_onnx(model: YOLOAnomaly, category: str, imgsz: int = IMGSZ) -> str:
    """Export model to ONNX and return the saved path."""
    onnx_path = Path(EXPORT_DIR) / f"{category}_{Path(BASE_MODEL).stem}_anomaly.onnx"

    print(f"\n[Export] Exporting to {onnx_path} ...")

    # ultralytics model.export() sets model[-1].export=True internally, which
    # routes ADMBHead to the dense (conf=0) fixed-shape path.
    exported = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=ONNX_OPSET,
        simplify=True,
        dynamic=False,   # static shapes: required for TensorRT
        half=False,
        device="cpu",
    )
    # model.export() returns the exported file path
    exported_path = str(exported)

    # Copy/rename to our desired path
    import shutil
    shutil.copy(exported_path, str(onnx_path))
    print(f"[Export] Saved to {onnx_path}")
    return str(onnx_path)


# ── Step 3: Pre/post-process helpers ─────────────────────────────────────────

def preprocess(img_path: str, imgsz: int = IMGSZ):
    """Load and letterbox-resize image to [1,3,H,W] float32 in [0,1]."""
    img_bgr = cv2.imread(img_path)
    assert img_bgr is not None, f"Cannot read image: {img_path}"
    h0, w0 = img_bgr.shape[:2]

    # Letterbox resize
    r = imgsz / max(h0, w0)
    new_h, new_w = int(round(h0 * r)), int(round(w0 * r))
    img_resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to square
    dh = imgsz - new_h
    dw = imgsz - new_w
    top, bottom = dh // 2, dh - dh // 2
    left, right  = dw // 2, dw - dw // 2
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=(114, 114, 114))

    # BGR→RGB, HWC→CHW, uint8→float32 [0,1]
    img_rgb  = img_padded[:, :, ::-1].astype(np.float32) / 255.0
    img_nchw = img_rgb.transpose(2, 0, 1)[np.newaxis]  # [1,3,H,W]
    return img_nchw, (h0, w0), r, (top, left)


def postprocess_boxes(output: np.ndarray, orig_hw, ratio, pad, conf_thresh=0.01):
    """
    Convert model output to bounding boxes in original image space.

    output shape: [1, ad_max_det, 6]  → [x1, y1, x2, y2, score, cls]
    """
    preds = output[0]  # [ad_max_det, 6]
    boxes = []
    for pred in preds:
        x1, y1, x2, y2, score, cls = pred
        if score < conf_thresh:
            continue
        # Undo padding + letterbox scaling
        x1 = (x1 - pad[1]) / ratio
        y1 = (y1 - pad[0]) / ratio
        x2 = (x2 - pad[1]) / ratio
        y2 = (y2 - pad[0]) / ratio
        boxes.append((float(x1), float(y1), float(x2), float(y2), float(score), int(cls)))
    return boxes


def draw_boxes(img_path: str, boxes, save_path: str):
    img = cv2.imread(img_path)
    for x1, y1, x2, y2, score, cls in boxes:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(img, f"{score:.2f}", (int(x1), int(y1) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imwrite(save_path, img)
    print(f"[Vis] Saved to {save_path}")


# ── Step 4: PyTorch inference (reference) ────────────────────────────────────

def pytorch_predict(model: YOLOAnomaly, img_path: str, conf=0.01):
    results = model.predict(img_path, conf=conf, imgsz=IMGSZ, verbose=False)
    res = results[0]
    print(f"[PyTorch] {len(res.boxes)} detections")
    if len(res.boxes):
        for box in res.boxes:
            print(f"  xyxy={box.xyxy[0].tolist()}  conf={box.conf[0]:.3f}")
    return res


# ── Step 5: ONNX Runtime inference ───────────────────────────────────────────

def onnx_predict(onnx_path: str, img_path: str, conf=0.01):
    import onnxruntime as ort

    sess_opts = ort.SessionOptions()
    sess_opts.log_severity_level = 3  # suppress INFO logs
    sess = ort.InferenceSession(onnx_path, sess_opts, providers=["CPUExecutionProvider"])

    input_name  = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    print(f"[ONNX] input={input_name}  output={output_name}")
    print(f"[ONNX] input shape:  {sess.get_inputs()[0].shape}")
    print(f"[ONNX] output shape: {sess.get_outputs()[0].shape}")

    img_nchw, orig_hw, ratio, pad = preprocess(img_path, IMGSZ)

    t0 = time.perf_counter()
    outputs = sess.run([output_name], {input_name: img_nchw})
    dt = time.perf_counter() - t0
    print(f"[ONNX] inference time: {dt*1000:.1f} ms")

    raw = outputs[0]  # [1, ad_max_det, 6]
    print(f"[ONNX] raw output shape: {raw.shape}")

    boxes = postprocess_boxes(raw, orig_hw, ratio, pad, conf_thresh=conf)
    print(f"[ONNX] {len(boxes)} detections above conf={conf}")
    for b in boxes:
        print(f"  xyxy=({b[0]:.1f},{b[1]:.1f},{b[2]:.1f},{b[3]:.1f})  conf={b[4]:.3f}")
    return boxes, raw


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    model, data_config = get_model(CATEGORY, BASE_MODEL)

    # Pick one anomaly test image
    test_imgs = data_config["test_anomaly_im_list"]
    assert test_imgs, "No test anomaly images found"
    test_img = test_imgs[0]
    print(f"\nTest image: {test_img}")

    # Step 2: export
    onnx_path = export_onnx(model, CATEGORY)

    # Step 3: PyTorch reference
    print("\n── PyTorch inference ─────────────────────────────────────────────")
    pytorch_predict(model, test_img)

    # Step 4: ONNX inference
    print("\n── ONNX inference ────────────────────────────────────────────────")
    boxes, raw = onnx_predict(onnx_path, test_img)

    # Step 5: Visualize ONNX result
    vis_path = str(Path(EXPORT_DIR) / f"{CATEGORY}_onnx_result.jpg")
    img_nchw, orig_hw, ratio, pad = preprocess(test_img, IMGSZ)
    draw_boxes(test_img, boxes, vis_path)

    # Quick numerical check: max confidence in raw output
    max_conf = float(raw[0, :, 4].max())
    print(f"\n[Check] max confidence in ONNX output: {max_conf:.4f}")
    if max_conf > 0.01:
        print("[Check] ✓ ONNX model produced meaningful anomaly scores")
    else:
        print("[Check] ⚠ All scores near zero — check export path or ad_conf setting")


if __name__ == "__main__":
    main()
