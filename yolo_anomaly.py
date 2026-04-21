import os
from pathlib import Path
from ultralytics.models.yolo.model import YOLOAnomaly
from ultra_ext.utils import open_in_vscode
from ultra_ext.im import vertical_concat_images
from time import sleep

def collect_images(directory, exts=(".png", ".jpg", ".jpeg"), max_n=None):
	"""Collect image paths from a directory (recursive), sorted, optionally capped."""
	imgs = sorted(
		str(p) for p in Path(directory).rglob("*") if p.suffix.lower() in exts
	)
	return imgs[:max_n] if max_n else imgs


def get_gt_mask(test_img):
	"""Return ground-truth mask path for an MVTec test image, or None."""
	mask = test_img.replace("test", "ground_truth").replace(".png", "_mask.png")
	return mask if os.path.exists(mask) else None


def main():
	# ── Config ────────────────────────────────────────────────────────────
	category = "leather"
	dataset_root = f"/Users/louis/workspace/public_datasets/mvtec_anomaly_detection/{category}"
	train_dir = f"{dataset_root}/train/good"
	test_dir = f"{dataset_root}/test"

	base_weights = "yoloe-11m-seg.pt"
	# base_weights= "../demos/fabric/best.pt"

	saved_model_path = f"./runs/temp/{category}_anomaly_model.pt"
	os.makedirs("./runs/temp", exist_ok=True)

	model_arg = dict(conf=0.1, iou=0.001, max_det=1000, imgsz=640)

	# # debug ,remove saved_model_path
	# if os.path.exists(saved_model_path):
	# 	os.remove(saved_model_path)

	# ── Build or load model ───────────────────────────────────────────────
	if os.path.exists(saved_model_path):
		model = YOLOAnomaly(saved_model_path)
		print(f"Loaded anomaly model from {saved_model_path}, is_configured={model.is_configured}")
	else:
		model = YOLOAnomaly(base_weights)
		model.setup(names=["anomaly"])
		model.set_ad_params(accumulate_thresh=0.1,score_filter_kernel=1)
		support_images = collect_images(train_dir, max_n=100)
		model.load_support_set(support_images, imgsz=model_arg["imgsz"])
		model.save(saved_model_path)
		print(f"Saved anomaly model to: {saved_model_path}")

	model.set_ad_params(ad_conf=0.5, ad_max_det=9, mode="anomaly" )

	# --- val -------------------

	model.val(data="/Users/louis/workspace/ultra_louis_work/buffer/MVTEC/MVTec-YOLO/leather.yaml",single_cls=True,visualize=True,plots=True,conf=0.5)

	# # ── Inference ─────────────────────────────────────────────────────────
	# test_imgs = collect_images(test_dir)
	# assert test_imgs, f"No test images found under {test_dir}"


	# for test_img in test_imgs[20:]:  # warmup / benchmark
	# 	res = model.predict([test_img], **model_arg)[0]
	# 	res.save("./runs/temp/res.png")
	# 	print(f"Predicted {len(res)} anomalies in {test_img} with conf>{model_arg['conf']}")

	# 	vertical_concat_images(
	# 		[test_img, get_gt_mask(test_img), "./runs/temp/res.png"],
	# 		texts=["Test Image", "Ground Truth Mask", "AD Result"],
	# 		save_path="./runs/temp/concat.png", layout="other",
	# 	)
	# 	open_in_vscode("./runs/temp/concat.png")
		
	# 	sleep(2)


main()
