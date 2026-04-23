import os
from pathlib import Path
from ultralytics.models.yolo.model import YOLOAnomaly
from ultra_ext.utils import open_in_vscode
from ultra_ext.im import vertical_concat_images
from time import sleep

def collect_images(directory, exts=(".png", ".jpg", ".jpeg",".bmp"), max_n=None, recursively=True):
	"""Collect image paths from a directory (recursive), sorted, optionally capped."""
	if recursively:
		imgs = sorted(
			str(p) for p in Path(directory).rglob("*") if p.suffix.lower() in exts
		)
	else:
		imgs = sorted(
			str(p) for p in Path(directory).glob("*") if p.suffix.lower() in exts
		)
	return imgs[:max_n] if max_n else imgs


def get_gt_mask(test_img):
	"""Return ground-truth mask path for an MVTec test image, or None."""
	mask = test_img.replace("test", "ground_truth").replace(".png", "_mask.png")
	return mask if os.path.exists(mask) else None

MVTEC_CATEGORIES = ["leather","grid","tile","wood","carpet",
				   "cable","hazelnut","pill","screw", # "toothbrush",
				   "metal_nut","capsule","bottle","transistor","zipper"]

def get_mvtec_yolo_data(category,data_root="/Users/louis/workspace/ultra_louis_work/buffer/MVTEC/MVTec-YOLO"):
	"""Return train/test image paths for a given MVTec category."""

	assert category in MVTEC_CATEGORIES, f"Unknown category: {category}"
	train_im_dir = f"{data_root}/{category}/train/good"
	test_im_dir = f"{data_root}/{category}/test"
	test_good_im_dir = f"{test_im_dir}/good"
	train_im_list=collect_images(train_im_dir,recursively=True)
	test_im_list=collect_images(test_im_dir,recursively=True)
	test_good_im_list=collect_images(test_good_im_dir,recursively=True)

	test_anomaly_im_list = [im for im in test_im_list if im not in test_good_im_list]	


	return dict(
		train_im_dir=train_im_dir,
		test_im_dir=test_im_dir,
		train_im_list=train_im_list,
		test_im_list=test_im_list,
		test_good_im_list=test_good_im_list,
		test_anomaly_im_list=test_anomaly_im_list,

		data_yaml=data_root+f"/{category}.yaml",
	)




def build_ad_model(base_weight, data_config, model_arg, anomaly_arg, category="screw",replace_model=False):
	base_weight_name=Path(base_weight).stem
	saved_model_path = f"./runs/temp/{category}_{base_weight_name}_anomaly_model.pt"
	# ── Build or load model ───────────────────────────────────────────────
	if not os.path.exists(saved_model_path) or replace_model:

		os.makedirs("./runs/temp", exist_ok=True)
		model = YOLOAnomaly(base_weight)
		model.setup(names=["anomaly"])
		model.set_ad_params(**anomaly_arg)
		support_images = collect_images(data_config["train_im_dir"])
		model.load_support_set(support_images, imgsz=model_arg["imgsz"])
		model.save(saved_model_path)
		print(f"Saved anomaly model to: {saved_model_path}")
	else:
		model = YOLOAnomaly(saved_model_path)
		print(f"Loaded anomaly model from {saved_model_path}, is_configured={model.is_configured}")
	return model

def iter_predict(data_config, model, model_arg,):
		test_imgs = data_config["test_anomaly_im_list"]



		for test_img in test_imgs[20:]:  # warmup / benchmark
			res = model.predict([test_img], **model_arg)[0]
			res.save("./runs/temp/res.png")
			print(f"Predicted {len(res)} anomalies in {test_img} with conf>{model_arg['conf']}")

			vertical_concat_images(
				[test_img, get_gt_mask(test_img), "./runs/temp/res.png"],
				texts=["Test Image", "Ground Truth Mask", "AD Result"],
				save_path="./runs/temp/concat.png", layout="other",
			)
			open_in_vscode("./runs/temp/concat.png")
			sleep(2)

def get_arguments(category="screw"):

	# if category =="leather":
	# 	model_arg = dict(conf=0.001, iou=0.001, max_det=1000, imgsz=640,single_cls=True,rect=False)
	# 	anomaly_arg=dict(accumulate_thresh=0.1,score_filter_kernel=0.1,ad_conf=0.5, ad_max_det=9, mode="anomaly")

	# if category =="grid":
	# 	model_arg = dict(conf=0.001, iou=0.001, max_det=1000, imgsz=640,single_cls=True,rect=False)
	# 	anomaly_arg=dict(accumulate_thresh=0.2,score_filter_kernel=0.1,ad_conf=0.4, ad_max_det=9, mode="anomaly")	

	model_arg = dict(conf=0.001, iou=0.001, max_det=1000, imgsz=640,single_cls=True,rect=False)
	anomaly_arg=dict(accumulate_thresh=0.3,score_filter_kernel=0.1,ad_conf=0.4, ad_max_det=9, mode="anomaly",
					 active_layers=[1, 2])  # 0=P3(fine), 1=P4, 2=P5(coarse) — only build/use selected layers
	
	return model_arg, anomaly_arg


def save_metrics_to_csv(metrics, save_path="./runs/temp/metrics.csv"):
	"""
	create the dir and csv if not exist, then save the metrics dict to csv, appending if file exists
	"""
	import csv
	import os

	os.makedirs(os.path.dirname(save_path), exist_ok=True)
	file_exists = os.path.isfile(save_path)

	with open(save_path, mode='a', newline='') as csv_file:
		writer = csv.DictWriter(csv_file, fieldnames=metrics.keys())
		if not file_exists:
			writer.writeheader()
		writer.writerow(metrics)
	print(f"Saved metrics to {save_path}")


def main():
	# ── Config ────────────────────────────────────────────────────────────
	categories=MVTEC_CATEGORIES

	# categories=["wood"]


	for category in categories:
		data_config=get_mvtec_yolo_data(category)

		base_model = "yoloe-11m-seg.pt"
		# base_model = "yoloe-26m-seg.pt"
		base_model="yolo26l.pt"
		model_arg, anomaly_arg = get_arguments(category)

		model = build_ad_model(base_model, data_config, model_arg, anomaly_arg, category=category, replace_model=True)

		# anomaly_arg=dict(accumulate_thresh=0.1,score_filter_kernel=0.1,ad_conf=0.4, ad_max_det=9, mode="anomaly")


		model.set_ad_params(**anomaly_arg)

		# --- val -------------------
		if True:
			model.set_ad_params(active_layers=[1, 2])
			res = model.val(data=data_config["data_yaml"], split="val", plots=False, batch=1, **model_arg)
			box = res.box
			metrics = dict(
				category=category,
				base_model=base_model,
				precision=float(box.mp),
				recall=float(box.mr),
				ap50=float(box.map50),
				ap50_95=float(box.map),
			)
			print(f"P={box.mp:.3f} R={box.mr:.3f} mAP50={box.map50:.3f} mAP50-95={box.map:.3f}")
			save_metrics_to_csv(metrics, save_path=f"./runs/temp/metrics_26l.csv")
		# ── Inference ─────────────────────────────────────────────────────────
		if False:
			iter_predict(data_config, model, model_arg)


if __name__ == "__main__":

	main()
