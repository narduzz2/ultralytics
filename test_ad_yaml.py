"""Test script for AnomalyDetection YAML parsing and save/load round-trip."""
import torch
import tempfile
import os
from ultralytics.nn.tasks import DetectionModel, load_checkpoint
from ultralytics.nn.modules.head import AnomalyDetection, ADMBHead

# Test 1: YAML parsing
print("=== Test 1: YAML parsing ===")
m = DetectionModel("ultralytics/cfg/models/26/yoloe-26-ad.yaml", verbose=False)
head = m.model[-1]
print(f"AnomalyDetection: {isinstance(head, AnomalyDetection)}, adhead: {head.adhead is not None}, nc={head.nc}")
assert isinstance(head, AnomalyDetection)
assert head.adhead is not None
assert head.nc == 1
print("PASS\n")

# Test 2: Save and reload model with populated memory bank
print("=== Test 2: Save/Load round-trip ===")
for h in head.adhead:
    if isinstance(h, ADMBHead):
        fake_dim = h.vocab_linear.in_features
        fake_mb = torch.randn(100, fake_dim)
        h.memory_bank = torch.nn.functional.normalize(fake_mb, p=2, dim=1)
        h.feature_dim = fake_dim
        h.update = False

tmpfile = os.path.join(tempfile.gettempdir(), "test_ad_model.pt")
ckpt = {
    "model": m.half(),
    "anomaly_meta": {
        "anomaly_names": {0: "anomaly"},
        "original_names": {},
        "original_nc": 1,
        "anomaly_mode": True,
        "ad_conf": 0.3,
        "ad_max_det": 5,
        "temperature": 3.0,
        "K": 15,
        "accumulate_thresh": 0.4,
    },
    "train_args": {"task": "detect"},
}
torch.save(ckpt, tmpfile)
print(f"Saved to {tmpfile}")

loaded_model, loaded_ckpt = load_checkpoint(tmpfile)
loaded_head = loaded_model.model[-1]
print(f"Loaded head type: {type(loaded_head).__name__}")
assert isinstance(loaded_head, AnomalyDetection)
assert loaded_head.adhead is not None
for i, h in enumerate(loaded_head.adhead):
    if isinstance(h, ADMBHead):
        print(f"  ADMBHead[{i}]: mb_shape={h.memory_bank.shape}")
        assert h.memory_bank.shape[0] == 100
assert "anomaly_meta" in loaded_ckpt
assert loaded_ckpt["anomaly_meta"]["ad_conf"] == 0.3
os.remove(tmpfile)
print("PASS\n")

# Test 3: YOLOAnomaly with pre-configured model
print("=== Test 3: YOLOAnomaly init with pre-configured model ===")
tmpfile2 = os.path.join(tempfile.gettempdir(), "test_ad_model2.pt")
torch.save(ckpt, tmpfile2)

from ultralytics.models.yolo.model import YOLOAnomaly
model = YOLOAnomaly(tmpfile2, verbose=False)
print(f"is_configured: {model.is_configured}")
assert model.is_configured
print(f"model.model.names: {model.model.names}")
os.remove(tmpfile2)
print("PASS\n")

print("=== All tests passed! ===")
