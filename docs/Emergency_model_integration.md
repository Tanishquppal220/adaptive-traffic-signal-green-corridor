# YOLOv8 Emergency Model Integration Guide

## 1. Project Goal

This model is trained to detect 7 object classes from the custom Roboflow dataset and then integrated into a larger system where:

- class IDs `3` and `5` are treated as **Emergency**
- all other class IDs are treated as **Non-Emergency**
- non-target detections can be visually masked in the output

---

## 2. Dataset Summary

### Source

- Dataset format: YOLO detection format
- Config file: `data/data.yaml`
- Declared split folders:
  - `train/images`
  - `valid/images`
  - `test/images`
- Number of classes: `7`
- Class names in the source YAML: `['0', '1', '2', '3', '4', '5', '6']`

### Roboflow metadata (from YAML)

- Workspace: `ambulance-b8kgx`
- Project: `ambulance-detection-tnzha`
- Version: `3`
- License: `CC BY 4.0`

### Colab path handling used

Training pipeline expects data from Google Drive and stages it in Colab:

- Drive dataset folder (preferred):
  - `/content/drive/MyDrive/emergency_workbench/data`
- Drive zip fallback:
  - `/content/drive/MyDrive/emergency_workbench/data.zip`
- Working dataset location in Colab:
  - `/content/emergency_yolo/data`

If the dataset folder is missing in Colab, the notebook copies from Drive folder, or unzips `data.zip` automatically.

---

## 3. Training Method Summary

### Base model

- Pretrained checkpoint: `yolov8n.pt`
- Task: object detection fine-tuning

### Training hyperparameters used

- Epochs: `50`
- Image size: `640`
- Batch size: `16`
- Early stopping patience: `10`
- Device: `GPU` (`device=0` in Colab)

### Subset training optimization

To reduce training time, a subset dataset is created before training:

- Train: `1000` images
- Validation: `200` images
- Test: `200` images
- Random seed: `42`

The notebook writes a subset YAML and points training to it:

- Subset data: `/content/emergency_yolo/data_subset`
- Subset YAML: `/content/emergency_yolo/data_colab_subset.yaml`

### Artifacts saved

- Best model: `/content/drive/MyDrive/emergency_yolo_artifacts/best_yolov8n_emergency.pt`
- Last model: `/content/drive/MyDrive/emergency_yolo_artifacts/last_yolov8n_emergency.pt`
- Final export snapshot + zip via notebook final export cell

---

## 4. Inference Integration Strategy

## 4.1 Core rule mapping

Define emergency classes once and reuse everywhere:

```python
EMERGENCY_CLASS_IDS = {3, 5}
```

Decision policy:

- If at least one detection in `EMERGENCY_CLASS_IDS` passes confidence threshold -> `EMERGENCY`
- Otherwise -> `NON_EMERGENCY`

## 4.2 Minimal inference code (single image)

```python
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = Path("/content/drive/MyDrive/emergency_yolo_artifacts/best_yolov8n_emergency.pt")
EMERGENCY_CLASS_IDS = {3, 5}
CONF_THRES = 0.25

model = YOLO(str(MODEL_PATH))

def infer_emergency(image_path: str):
    results = model.predict(source=image_path, conf=CONF_THRES, verbose=False)
    if not results:
        return {
            "decision": "NON_EMERGENCY",
            "reason": "No detections",
            "detections": []
        }

    r = results[0]
    detections = []
    has_emergency = False

    if r.boxes is not None and len(r.boxes) > 0:
        for b in r.boxes:
            cls_id = int(b.cls.item())
            conf = float(b.conf.item())
            xyxy = [float(v) for v in b.xyxy[0].tolist()]

            is_emergency = cls_id in EMERGENCY_CLASS_IDS
            has_emergency = has_emergency or is_emergency

            detections.append({
                "class_id": cls_id,
                "class_name": str(model.names.get(cls_id, cls_id)),
                "confidence": conf,
                "bbox_xyxy": xyxy,
                "is_emergency": is_emergency,
            })

    return {
        "decision": "EMERGENCY" if has_emergency else "NON_EMERGENCY",
        "detections": detections,
    }
```

---

## 5. Mask All Non-Target Classes (keep only 3 and 5 visible)

Use this when you want output images where non-emergency objects are hidden.

```python
import cv2
import numpy as np

EMERGENCY_CLASS_IDS = {3, 5}

def mask_non_emergency_regions(image_bgr: np.ndarray, result, emergency_ids=EMERGENCY_CLASS_IDS):
    """
    Keeps only emergency detections visible.
    Non-emergency detection boxes are blacked out.
    """
    out = image_bgr.copy()

    if result.boxes is None or len(result.boxes) == 0:
        return out

    for b in result.boxes:
        cls_id = int(b.cls.item())
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]

        if cls_id not in emergency_ids:
            out[y1:y2, x1:x2] = 0  # black mask

    return out

# Example usage
# image_bgr = cv2.imread("input.jpg")
# res = model.predict(source=image_bgr, conf=0.25, verbose=False)[0]
# masked = mask_non_emergency_regions(image_bgr, res)
# cv2.imwrite("masked_output.jpg", masked)
```

Optional alternative mask styles:

- blur non-emergency boxes instead of black fill
- dim with alpha overlay
- draw only emergency boxes and hide everything else

---

## 6. Batch Inference for Production Pipeline

```python
from pathlib import Path
import json

INPUT_DIR = Path("/path/to/input_images")
OUTPUT_JSON = Path("/path/to/inference_results.json")

all_results = []
for img_path in INPUT_DIR.glob("*.*"):
    if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        continue

    pred = infer_emergency(str(img_path))
    pred["image"] = img_path.name
    all_results.append(pred)

OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
with OUTPUT_JSON.open("w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2)
```

---

## 7. Suggested Integration Contract

Return a stable payload to your larger project:

```json
{
  "decision": "EMERGENCY",
  "detections": [
    {
      "class_id": 5,
      "class_name": "5",
      "confidence": 0.91,
      "bbox_xyxy": [114.5, 60.2, 340.8, 280.1],
      "is_emergency": true
    }
  ]
}
```

Recommended fields:

- `decision`: `EMERGENCY` or `NON_EMERGENCY`
- `detections`: full detection list for traceability
- `is_emergency` per detection for downstream filtering

---

## 8. Practical Notes Before Deployment

- Verify class ID mapping once from the final trained model (`model.names`), then lock it in config.
- Keep confidence threshold configurable (start with `0.25`, tune using validation data).
- Log both:
  - final decision
  - top emergency confidence
- Save sample failure cases (false positives and false negatives) for future retraining.

---

## 9. Recommended Next Step

For robust production behavior, add a small configuration file in your main project:

```yaml
model_path: /models/best_yolov8n_emergency.pt
confidence_threshold: 0.25
emergency_class_ids: [3, 5]
mask_non_emergency: true
```

Then load this config in your inference service so class logic can be changed without code edits.
