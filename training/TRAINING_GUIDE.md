# Traffic Detection Model Training Guide

## Overview
This guide covers training an efficient YOLOv8 nano model on 500 selected traffic vehicle detection images with minimal resource requirements.

## Prerequisites

### System Requirements
- **GPU**: Optional (CUDA 11.8+ recommended, but CPU works)
- **RAM**: Minimum 4GB, recommended 8GB+
- **Storage**: ~2GB (model weights + dataset)

### Python Environment
```bash
# Install required packages
pip install ultralytics>=8.0.0
pip install torch>=2.0.0
pip install pyyaml>=6.0
pip install numpy>=1.21.0
pip install pillow>=9.0.0
pip install opencv-python>=4.6.0
```

Or use the project setup:
```bash
cd /workspaces/adaptive-traffic-signal-green-corridor
pip install -e .
```

## Dataset Setup

The notebook **automatically**:
1. Locates the traffic dataset at `data/raw/traffic-vehicles-object-detection/Traffic Dataset`
2. Selects 500 random images from the training set (738 total)
3. Copies images and YOLO-format labels to `data/processed/traffic_subset_500`
4. Splits into 90% train / 10% validation
5. Creates `data.yaml` configuration for YOLO

### Dataset Structure
```
data/raw/traffic-vehicles-object-detection/Traffic Dataset/
├── images/
│   ├── train/          (738 images)
│   ├── val/
│   └── test/
└── labels/
    └── train/          (738 label files in YOLO format)
```

## Running the Training

### Option 1: Jupyter Notebook (Recommended)
```bash
jupyter notebook training/trafficDetection.ipynb
```

Then execute cells in order (Shift+Enter):
1. **Cell 1-2**: Imports and setup
2. **Cell 3-4**: Dataset configuration and path setup
3. **Cell 5**: Select 500 images and prepare dataset
4. **Cell 6-7**: Create YOLO configuration
5. **Cell 8-9**: Load model and train
6. **Cell 10**: Run evaluation
7. **Cell 11**: Agentic evaluation loop (self-critique)
8. **Cell 12**: Test on sample images

### Option 2: Command Line Training
```python
from ultralytics import YOLO
from pathlib import Path

# Load model
model = YOLO('yolov8n.pt')

# Train
results = model.train(
    data='data/processed/traffic_subset_500/data.yaml',
    epochs=25,
    imgsz=416,
    batch=8,
    device=0,  # Use GPU (or 'cpu')
    patience=5
)
```

## Training Performance

### Resource Usage
| Metric | Value |
|--------|-------|
| Model Size | ~6 MB |
| Memory Peak | ~500 MB (GPU) / ~1 GB (CPU) |
| Training Time (GPU) | ~3-5 minutes |
| Training Time (CPU) | ~10-15 minutes |
| Images per Epoch | 450 training + 50 validation |

### Expected Results
- **mAP50**: 0.70-0.85 (typical range)
- **mAP**: 0.45-0.60 (0-100 IoU)
- **Inference Speed**: 30-50 ms per image

## Agentic Evaluation Loop

The notebook includes a **self-critique evaluation pattern** that:

1. **Evaluates** the model against metrics (mAP50, mAP)
2. **Critiques** performance against a threshold (0.70)
3. **Suggests** refinements if needed
4. **Tracks** improvement across iterations

Example output:
```
[Iteration 1/2]
  mAP50: 0.7234
  mAP:   0.5108
  Score: 0.6511
  Critique: ⚠ Nearly there: Close to threshold. Slight improvements needed.
  Suggestion: Fine-grained optimization with lower learning rate

✓ Model improvement completed: threshold reached!
```

## Model Output

### Saved Model Location
```
models/traffic_detection_yolov8n.pt
```

### Using the Trained Model

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('models/traffic_detection_yolov8n.pt')

# Predict on image
results = model.predict(image_path, conf=0.5)

# Get vehicle count
vehicle_count = len(results[0].boxes)

# Get predictions with confidence
for box in results[0].boxes:
    confidence = float(box.conf)
    xyxy = box.xyxy[0]  # coordinates
    print(f"Vehicle detected: confidence={confidence:.2%}")
```

## Optimization Tips

### For Better Accuracy
- Train longer: Increase `EPOCHS` from 25 to 50-100
- Larger images: Change `IMG_SIZE` from 416 to 640
- More data: Use all 738 images or increase `num_samples` to 600-700

### For Faster Training
- Reduce batch size: Change `BATCH_SIZE` from 8 to 4 (trades memory for speed)
- Smaller model: Use `yolov8n` (nano) - already used
- Reduce augmentation: Lower HSV/flip probabilities

### For Production Deployment
- Export model: `model.export(format='onnx')` for cross-platform
- Quantize: Use INT8 quantization for mobile/edge devices
- Batch processing: Process multiple images simultaneously

## Troubleshooting

### Out of Memory
```python
# Reduce batch size
BATCH_SIZE = 4

# Reduce image size
IMG_SIZE = 320
```

### Poor Accuracy
- Check label files are in correct YOLO format
- Verify at least 500 images have corresponding labels
- Increase epochs or dataset size

### CUDA Errors
```python
# Use CPU instead
device = 'cpu'

# Or specify specific GPU
device = 1  # for second GPU
```

## References
- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com)
- [YOLO Label Format](https://docs.ultralytics.com/datasets/detect)
- [Agentic Evaluation Patterns](./../.github/skills/agentic-eval/SKILL.md)

## Next Steps

After training:
1. Integrate model into `detection/` module
2. Add inference function to `control/` module
3. Create integration tests
4. Monitor model performance in production
