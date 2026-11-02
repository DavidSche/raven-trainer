"""Export best.pt to ONNX format and verify."""
import shutil
from pathlib import Path
from ultralytics import YOLO

base = Path(r"E:\study-place\Davidche\2026\raven\raven-trainer")

# Source weights
best_pt = base / "projects/d6_person_intrusion/runs/yolo26s_20260731_112953/weights/best.pt"
print(f"Source weights: {best_pt}")
print(f"Exists: {best_pt.exists()}")

if not best_pt.exists():
    print("ERROR: best.pt not found!")
    exit(1)

# Load model
model = YOLO(str(best_pt))
print(f"Model loaded: {model.model_name if hasattr(model, 'model_name') else 'yolo26s'}")

# Export to ONNX
print("\n[1/3] Exporting to ONNX...")
onnx_path = model.export(
    format="onnx",
    opset=17,
    dynamic=True,
    simplify=True,
    imgsz=640,
)
print(f"  ONNX exported: {onnx_path}")

# Copy to models directory
models_dir = base / "projects/d6_person_intrusion/models"
models_dir.mkdir(parents=True, exist_ok=True)

dest_onnx = models_dir / "v1.0.onnx"
dest_pt = models_dir / "v1.0.pt"

print("\n[2/3] Copying weights to models directory...")
shutil.copy2(best_pt, dest_pt)
shutil.copy2(onnx_path, dest_onnx)

print(f"  PT:   {dest_pt} ({dest_pt.stat().st_size/1024/1024:.1f} MB)")
print(f"  ONNX: {dest_onnx} ({dest_onnx.stat().st_size/1024/1024:.1f} MB)")

# Verify ONNX
print("\n[3/3] Verifying ONNX model...")
try:
    import onnx
    onnx_model = onnx.load(str(dest_onnx))
    onnx.checker.check_model(onnx_model)
    print("  ONNX model is valid!")
    
    # Print input/output info
    for inp in onnx_model.graph.input:
        shape = [d.dim_value if d.dim_value > 0 else d.dim_param for d in inp.type.tensor_type.shape.dim]
        print(f"  Input: {inp.name} shape={shape}")
    for out in onnx_model.graph.output:
        shape = [d.dim_value if d.dim_value > 0 else d.dim_param for d in out.type.tensor_type.shape.dim]
        print(f"  Output: {out.name} shape={shape}")
except ImportError:
    print("  onnx package not installed, skipping verification")
except Exception as e:
    print(f"  Verification warning: {e}")

# Quick inference test
print("\n=== Quick Inference Test ===")
import cv2
import numpy as np

# Find a test image
val_dir = base / "data/workspace/d6_train_v1v2/val/images"
test_images = sorted(val_dir.glob("*.jpg"))[:3]

if test_images:
    for img_path in test_images:
        results = model.predict(str(img_path), conf=0.5, verbose=False)
        for r in results:
            boxes = r.boxes
            print(f"  {img_path.name}: {len(boxes)} detections")
            for box in boxes:
                conf = box.conf[0].item()
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                print(f"    person {conf:.3f} [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")

print("\n" + "=" * 60)
print("ONNX Export Complete!")
print("=" * 60)
