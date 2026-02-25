# Quick Start - Traffic Detection Training

## 🚀 Run in 3 Steps

### 1. Install Dependencies
```bash
cd /workspaces/adaptive-traffic-signal-green-corridor
pip install -e .
```

### 2. Open Notebook
```bash
jupyter notebook training/trafficDetection.ipynb
```

### 3. Execute Cells (Shift+Enter)
Execute all cells from top to bottom. Total time: **~10 minutes** (CPU) or **~3 minutes** (GPU).

---

## 📊 What You Get

| Output | Location | Purpose |
|--------|----------|---------|
| **Dataset** | `data/processed/traffic_subset_500/` | 500 images + labels for training |
| **Trained Model** | `models/traffic_detection_yolov8n.pt` | Ready-to-use YOLOv8n weights |
| **Training Logs** | `runs/detect/train/` | Metrics, graphs, weights |
| **Evaluation** | Console output | mAP50, mAP, improvement metrics |

---

## 💡 Key Details

**Model**: YOLOv8n (nano)
- Size: ~6 MB
- Speed: 30-50 ms per image
- Memory: 500 MB (GPU) / 1 GB (CPU)

**Dataset**: 500 randomly selected images
- Source: 738 training images
- Split: 450 train / 50 validation
- Format: YOLO (images + .txt labels)

**Training**:
- Epochs: 25 (tunable)
- Batch: 8 (smaller for less memory)
- Optimizer: SGD (faster than Adam)
- Device: Auto-detects GPU/CPU

**Features**:
- ✅ Self-critique evaluation loop
- ✅ Early stopping to prevent overfitting
- ✅ Automatic augmentation tuning
- ✅ Complete inference pipeline

---

## 🔍 First-Time Checklist

- [ ] Dataset exists: `data/raw/traffic-vehicles-object-detection/Traffic Dataset/`
- [ ] Python: `python --version` (3.13+)
- [ ] Dependencies: `pip list | grep ultralytics`
- [ ] Output directories will be created automatically

---

## 📖 Learn More

| Document | Focus |
|----------|-------|
| [TRAINING_GUIDE.md](TRAINING_GUIDE.md) | Complete training guide with troubleshooting |
| [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) | Architecture and design decisions |
| [INTEGRATION_EXAMPLES.md](INTEGRATION_EXAMPLES.md) | Code examples for using the model |

---

## ⚡ Common Issues & Fixes

### "Out of Memory"
```python
BATCH_SIZE = 4  # Reduce from 8
IMG_SIZE = 320  # Reduce from 416
```

### "CUDA not available"
```python
# Notebook automatically uses CPU if CUDA unavailable
# No code changes needed - runs at ~10-15 min on CPU
```

### "Dataset not found"
```python
# Verify this path exists:
# data/raw/traffic-vehicles-object-detection/Traffic Dataset/images/train/
```

### "Import errors"
```bash
pip install -e .  # Reinstall project
pip install --upgrade ultralytics
```

---

## 🎯 Next Steps After Training

1. **Verify Model**: Check `models/traffic_detection_yolov8n.pt` exists
2. **Test Inference**: Run Cell 16 (inference on samples)
3. **Integrate**: Copy `VehicleDetector` class from `INTEGRATION_EXAMPLES.md`
4. **Deploy**: Use in `detection/` module

---

## 📝 Monitor Progress

**During training**, check console for:
```
Epoch 1/25: loss=0.45, val_loss=0.38
Epoch 2/25: loss=0.38, val_loss=0.35
...
```

**After training**, expect:
```
✓ Evaluation completed
mAP50: 0.72 (typical range: 0.70-0.85)
mAP:   0.51 (typical range: 0.45-0.60)
```

---

## 📞 Support

- **YOLOv8 Docs**: https://docs.ultralytics.com
- **Dataset Issues**: Check `data/README.md`
- **Code Questions**: See `copilot-instructions.md`
- **Agentic Patterns**: Read `.github/skills/agentic-eval/SKILL.md`
