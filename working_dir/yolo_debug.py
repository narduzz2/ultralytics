from ultralytics import YOLO
from ultralytics import RTDETR, RTDETRDEIM
from ultralytics.utils import ROOT, YAML
import torch


# model = YOLO('working_dir/yolo26nms_weights/yolo26n_nms.onnx')
# model = RTDETR('onnx_exports/rfdetr-nano/rfdetr-nano.onnx')
# model = YOLO('yolo26n.onnx')
# model = YOLO('yolo26n.engine')

# model = YOLO('ultralytics/cfg/models/26/yolo26n_nms.yaml')
# model.load('yolo26n.pt')
# # model.save("working_dir/yolo26nms_weights/yolo26n_nms.pt")

# model = YOLO('ultralytics/cfg/models/26/yolo26s_nms.yaml')
# model.load('yolo26s.pt')
# # model.save("working_dir/yolo26nms_weights/yolo26s_nms.pt")

# model = YOLO('ultralytics/cfg/models/26/yolo26m_nms.yaml')
# model.load('yolo26m.pt')
# # model.save("working_dir/yolo26nms_weights/yolo26m_nms.pt")

# model = YOLO('ultralytics/cfg/models/26/yolo26l_nms.yaml')
# model.load('yolo26l.pt')
# # model.save("working_dir/yolo26nms_weights/yolo26l_nms.pt")

# model = YOLO('ultralytics/cfg/models/26/yolo26x_nms.yaml')
# model.load('yolo26x.pt')
# model.save("working_dir/yolo26nms_weights/yolo26x_nms.pt")

# model = YOLO("working_dir/yolo26nms_weights/yolo26n_nms.pt")
# model.export(format="onnx", nms=True)

# model = YOLO("working_dir/yolo26nms_weights/yolo26s_nms.pt")
# model.export(format="onnx", nms=True)

# model = YOLO("working_dir/yolo26nms_weights/yolo26m_nms.pt")
# model.export(format="onnx", nms=True)

# model = YOLO("working_dir/yolo26nms_weights/yolo26l_nms.pt")
# model.export(format="onnx", nms=True)

# model = YOLO("working_dir/yolo26nms_weights/yolo26x_nms.pt")
# model.export(format="onnx", nms=True)

# model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-l.yaml')
# model.load('rtdetr-l.pt')
# model = RTDETR('rtdetr-l.engine')

# model = RTDETR('working_dir/yolodetr_weights/yolo26_detr_l_obj_640.pt')

# Yolo26-rtdetr results
# model = RTDETR('ultralytics/cfg/models/26/yolo26l-rtdetr.yaml')
# model.load('/Users/esat/workspace/runs/rtdetr_yolo26l_PObj_origaugV2_imgsz640_epc90_clsmos15_lrf05/weights/best.pt')

# # Yolo26n-rtdetr results
# model = RTDETR('ultralytics/cfg/models/26/yolo26n-rtdetr_p4_l3_efms_365.yaml')
# model.load('/Users/esat/workspace/rtdetrLightp4_yolo26n_scratch_wu1_lr4x_origaugV2_150epc/weights/best.pt')

# # model = YOLO('ultralytics/cfg/models/26/yolo26n.yaml')
# model.load('/Users/esat/workspace/rtdetrLightp4_yolo26n_scratch_wu1_lr4x_origaugV2_150epc/weights/best.pt')

# model = RTDETR('ultralytics/cfg/models/26/yolo26n-rtdetr_p4_l3_efms.yaml')
# model.load('/Users/esat/workspace/pretrained/rtdetrLightp4_yolo26n_PObj_origaugV2_imgsz480_300epc/weights/best.pt')
# model.save("working_dir/yolodetr_weights/yolo26_detr_n_obj_480.pt")

# model = RTDETR('ultralytics/cfg/models/26/yolo26s-rtdetr_p4_l3_efms.yaml')
# model.load('/Users/esat/workspace/pretrained/rtdetrLight_yolo26s_PCoco_origaugV2_imgsz512_epc120/weights/best.pt')
# model.save("working_dir/yolodetr_weights/yolo26_detr_ns_coco_512.pt")
# model.load('/Users/esat/workspace/pretrained/rtdetr_yolo26ns_PObj_lrf05_origaugV2_clsmos15_imgsz512/weights/best.pt')
# model.save("working_dir/yolodetr_weights/yolo26_detr_ns_obj_512.pt")

# model = RTDETR('ultralytics/cfg/models/26/yolo26s-rtdetr.yaml')
# # model.load('/Users/esat/workspace/pretrained/rtdetr_yolo26s_PCoco_cnstLR_wu0_lr1x_origaugV2_imgsz640_epc120/weights/best.pt')
# # model.save("working_dir/yolodetr_weights/yolo26_detr_s_coco_640.pt")
# model.load('/Users/esat/workspace/pretrained/rtdetr_yolo26m_PObj_lrf05_origaugV2_clsmos15/weights/rtdetr_yolo26s_PObj_lrf05_origaugV2_clsmos15/weights/best.pt')
# model.save("working_dir/yolodetr_weights/yolo26_detr_s_obj_640.pt")

# model = RTDETR('ultralytics/cfg/models/26/yolo26m-rtdetr.yaml')
# model.load('/Users/esat/workspace/pretrained/rtdetr_yolo26m_PObj_lrf05_origaugV2_clsmos15/weights/best.pt')
# model.save("working_dir/yolodetr_weights/yolo26_detr_m_obj_640.pt")

# model = RTDETR('ultralytics/cfg/models/26/yolo26l-rtdetr.yaml')
# model.load('/Users/esat/workspace/pretrained/rtdetr_yolo26l_PObj_origaugV2_imgsz640_epc90_clsmos15_lrf05/weights/best.pt')
# model = RTDETR('/Users/esat/workspace/pretrained/rtdetr_yolo26l_PObj_origaugV2_FL_epc90_clsmos15_lrf05/weights/best.pt')
# model = RTDETR("working_dir/yolodetr_weights/yolo26_detr_l_obj_640.onnx")
# model = RTDETR("working_dir/onnx_exports/rfdetr-nano/rfdetr-nano.onnx")
# model.save("working_dir/yolodetr_weights/yolo26_detr_l_obj_640.pt")

# model = RTDETR('ultralytics/cfg/models/26/yolo26l-rtdetr_l4.yaml')
# model.load('/Users/esat/workspace/pretrained/rtdetr_l4_yolo26l_PObj_origaugV2_clsmos15_lrf05_imgsz704/weights/best.pt')
# model.save("working_dir/yolodetr_weights/yolo26_detrl4_l_obj_704.pt")

# model = RTDETR('ultralytics/cfg/models/26/yolo26x-rtdetr.yaml')
# # model.load('/Users/esat/workspace/pretrained/rtdetr_yolo26x_PCoco_v10_cnstLR_wu0_lr1x_origaugV2_imgsz640_epc90/weights/best.pt')
# # model.save("working_dir/yolodetr_weights/yolo26_detr_x_coco_640.pt")
# model.load('/Users/esat/workspace/pretrained/rtdetr_yolo26x_PObj_lrf05_origaugV2_clsmos15/weights/best.pt')
# model.save("working_dir/yolodetr_weights/yolo26_detr_x_obj_640.pt")

# model = RTDETR('ultralytics/cfg/models/26/yolo26l-rtdetr.yaml')
# model.load('/Users/esat/workspace/pretrained/rtdetr_yolo26l_PObj_origaugV2_imgsz640_epc90_clsmos15_lrf05/weights/best.pt')
# model.load('/Users/esat/workspace/pretrained/rtdetr_yolo26l_PObj_origaugV2_clsmos40_lrf05_mosp05/weights/last.pt')
# model.load('/Users/esat/workspace/pretrained/rtdetr_yolo26l_PObj_origaugV2_imgsz640_epc90_clsmos15_lrf05_mal/weights/last.pt')

# model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr_dinov3s_l6_sta.yaml')
# # model.load('/Users/esat/workspace/pretrained/rtdetr_dinov3sta_dec6_lrf05_origaugV2_clsmos15/weights/best.pt')
# model.load('working_dir/dinov3_weights/rtdetr_dinov3sta_detrl6_640.pt')
# # model.save("working_dir/dinov3_weights/rtdetr_dinov3sta_detrl6_640.pt")

# model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr_dinov3s_l3_sta.yaml')
# model.load('/Users/esat/workspace/pretrained/rtdetr_dinov3staL_dec3_lrf05_origaugV2_clsmos15/weights/best.pt')
# model.load('working_dir/dinov3_weights/rtdetr_dinov3sta_detrl3_640.pt')
# model.save("working_dir/dinov3_weights/rtdetr_dinov3sta_detrl3_640.pt")
# model = RTDETRDEIM("/data/esat/workspace/runs/detect/detr_trainings/deim_dinov3_v3_yoloaugp05_decay_68epc_obj2cocov2_epc24/weights/best.pt")
# model = RTDETRDEIM("/data/esat/workspace/runs/detect/detr_trainings/deim_dinov3_v3_yoloaugp05_decay_68epc_obj2cocov2_epc24/weights/best_op17_nosim_fp16_91new.engine")
# model = RTDETRDEIM("/data/esat/workspace/runs/detect/detr_trainings/deim_dinov3_v3_yoloaugp05_decay_68epc_obj2cocov2_epc24/weights/best_op17_nosim_norope_fp16.engine")
# model = RTDETRDEIM("/data/esat/workspace/runs/detect/detr_trainings/deim_dinov3plusSTA_xl_yoloaug_decay_58epc_p05_noaug4/weights/best_op17_nosim_norope_fp16working.engine")
model = RTDETRDEIM("/home/esat/workspace/runs/detect/detr_trainings/deim_dinov3plusSTA_xl_yoloaug_decay_58epc_p07_noaug4/weights/best_op17_nosim_norope_fp16.engine")
# model = RTDETRDEIM("/data/esat/workspace/runs/detect/detr_trainings/deim_dinov3baseSTA_xxl_yoloaug_decay_58epc_p07_noaug4/weights/best_op17_nosim_norope_fp16.engine")

# model = RTDETRDEIM("/home/esat/workspace/deimv2_ultralytics/deimv2XL_coco_v2_op17_nosim_norope_fp16.engine")
# model = RTDETRDEIM("/home/esat/workspace/deimv2_ultralytics/deimv2XXL_coco_op17_nosim_norope_fp16.engine")
# model = RTDETR("yolo26_detr_l_obj_640.engine")

# model = RTDETR("ultralytics/cfg/models/26/yolo26_rtdetr_dinov3s_l3_light_obj365.yaml")
# model = YOLO("ultralytics/cfg/models/26/yolo26_dinov3s_l3_light.yaml")
# model = RTDETR("/Users/esat/workspace/pretrained/rtdetr_l3_dinov3origaugV2_imgsz640_epc150_clsmos15_lrf05_lr4x_obj365/weights/best.pt")s
# model = RTDETR("ultralytics/cfg/models/26/yolo26_rtdetr_dinov3s_l3_light.yaml")
# weights = "/Users/esat/workspace/pretrained/rtdetr_l3_dinov3origaugV2_imgsz640_epc150_clsmos15_lrf05_lr4x_obj365/weights/best.pt"
# # weights = "dinov3_small_detr_pretrained_wo_decoder_weights.pt"
# weights = "/Users/esat/workspace/pretrained/rtdetr_l3_dinov3origaugV2_imgsz512_epc120_clsmos30_lrf05/weights/last.pt"
# obj365_names = YAML.load("working_dir/datasets/Objects365v1.yaml").get("names")
# model.model.dst_names = obj365_names
# model.load(weights)
# model.save("dinov3_small_detr_pretrained_wo_decoder_weights.pt")

# coco_yaml = ROOT / "cfg/datasets/coco.yaml"
# coco_names = YAML.load(coco_yaml).get("names")

# from ultralytics.nn.tasks import load_checkpoint
# model = RTDETR("ultralytics/cfg/models/rt-detr/rtdetr_dinov3s_l3_light.yaml")
# weights = "/Users/esat/workspace/pretrained/rtdetr_l3_dinov3origaugV2_imgsz640_epc150_clsmos15_lrf05_lr4x_obj365/weights/best.pt"
# src_model, ckpt = load_checkpoint(weights)
# model.ckpt = ckpt  # optional, keeps wrapper behavior parity
# model.model.load(
#     src_model,
#     src_names=getattr(src_model, "names", None),
#     dst_names=coco_names,
#     verbose=True,
# )

# model = RTDETR("/Users/esat/workspace/pretrained/rtdetrLightp4_yolo26n_scratch_wu1_lr4x_origaugV2_150epc/weights/best.pt")

# Load COCO class names from the dataset config
coco_yaml = ROOT / "cfg/datasets/coco.yaml"
coco_names = YAML.load(coco_yaml).get("names")
if hasattr(model.model, "names"):
    # PyTorch checkpoints expose the underlying nn.Module, so names can be set directly.
    model.model.names = coco_names
else:
    # Exported backends such as TensorRT keep model.model as a path string.
    # Point the predictor at COCO yaml so AutoBackend can populate class names during setup.
    model.overrides["data"] = str(coco_yaml)

# Load Obj class names from the dataset config
# obj365_names = YAML.load("working_dir/datasets/Objects365v1.yaml").get("names")
# model.model.names = obj365_names  # Set names on the underlying model object


# # ckpt = torch.load("yolo26s-objv1-150.pt", weights_only=False)
# ckpt = torch.load("yolo26s.pt", weights_only=False)
# # ckpt = torch.load("yolo26l.pt", weights_only=False)
# train_args = ckpt.get("train_args")
# print(ckpt["train_args"])
# model = RTDETR("rtdetr-l.pt")

# 2. Run inference on a source (can be a file path, URL, or '0' for webcam)
# We use a standard URL image for this example.
# model.set_head_attr(disable_topk=True)  # Disable top-k selection to get raw predictions for debugging
results = model('https://ultralytics.com/images/bus.jpg', conf=0.28, imgsz=640)

# 3. Iterate through results (usually a list, one per image)
for i, r in enumerate(results):
    print(f"\n--- Debugging Image {i+1} ---")

    # 'r.boxes' contains the detection data
    # .data gives you the raw tensor: [x1, y1, x2, y2, conf, class_id]
    print(f"Total Detections: {len(r.boxes)}")
    
    # Iterate over each box to print specific debug info
    for box in r.boxes:
        # Get coordinates (x1, y1, x2, y2)
        # .cpu().numpy() converts the tensor to a readable numpy array
        coords = box.xyxy[0].cpu().numpy() 
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        cls_name = r.names[cls_id] # Map ID to class name
        
        print(f"Object: {cls_name} | Conf: {conf:.2f} | Box: {coords}")

    # OPTIONAL: Visualize the result
    # r.show()  # Opens a window with the image
    r.save(filename=f'working_dir/results/result_{i}.jpg') # Saves the image to disk
    print(f"\nSaved visualization to 'working_dir/results/result_{i}.jpg'")
