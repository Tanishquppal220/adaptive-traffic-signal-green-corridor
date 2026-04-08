
## Features

- **Vehicle Detection** — YOLOv8s fine-tuned single-class detector counts vehicles per lane (N / S / E / W)
- **Emergency Vehicle Classification** — YOLOv8s classifier detects ambulances / fire trucks with calibrated confidence thresholds
- **Siren Detection** — TFLite model on raw WAV audio for siren / no-siren classification
- **Density Prediction** — XGBoost × 4 (per direction) forecasts 60-second-ahead traffic density from a 100-step sliding window
- **DQN Signal Optimiser** — 224-action Deep Q-Network (4 directions × 56 durations) for adaptive green-time allocation
- **Emergency Green Corridor** — Dual-gated emergency policy: activates only when **both** visual detection AND siren audio confirm an emergency
- **Online Learning** — DQN continues refining weights after each real cycle via experience replay

---

## ML Model Stack

### 1. Traffic Detection — `trafficDetection.ipynb`

- **Model:** YOLOv8s fine-tuned, single class `vehicle`
- **Input:** One image per direction (N / S / E / W)
- **Output:** `q_N, q_S, q_E, q_W` — vehicle count per lane (via bounding-box centre quadrant mapping)
- **Weights:** `models/traffic_detection_yolov8s.pt`
- **Runtime wrapper:** TODO 
- **Training config:** 80 epochs, img 736, batch 16, AdamW, freeze=10, early stop patience=15

### 2. Emergency Vehicle Classification — `emergnecy_detection.ipynb`

- **Model:** YOLOv8s fine-tuned on emergency vehicle dataset
- **Input:** Same 4 direction images (run after traffic detection)
- **Output:** `emergency_flag = 1` if target class IDs (3 or 5) detected above calibrated threshold
- **Weights:** `models/emergency_vehicle_cls_yolov8s.pt`
- **Threshold:** Auto-loaded from `models/emergency_classifier_threshold.json` (F1-optimal), fallback default 0.5
- **Runtime wrapper:** TODO
- **Training config:** 50 epochs, img 512, batch 16, AdamW, freeze=10, early stop patience=10

### 3. Siren Detection - `siren_detection.ipynb`

- **Model:** Custom TFLite binary classifier
- **Input:** 1 WAV audio file per upload session
- **Output:** `siren_flag = 1` if siren confidence ≥ threshold (default 0.5)
- **Weights:** `models/siren_detector_v2.tflite`
- **Audio pipeline:** WAV decode → mono conversion → linear resampling to 16 kHz → zero-pad to minimum duration → TFLite inference
- **Runtime wrapper:** TODO

### 4. Density Predictor — `traffic-Density.ipynb`

- **Model:** XGBoost Booster × 4 (one per direction N / S / E / W)
- **Input:** 100-step sliding window of past counts (400 floats) + 4 cyclic time features (sin/cos hour + sin/cos day-of-week) = **404-dim** feature vector
- **Output:** Predicted avg vehicle count 60 seconds ahead per direction
- **Weights:** `models/density_predictor_{N,S,E,W}.ubj`
- **Runtime wrapper:** TODO
- **Note:** Falls back to heuristic (latest counts) when history < 100 or models unavailable

### 5. DQN Signal Optimiser — `training/DQN/`

- **Framework:** PyTorch
- **State (6-dim):** `[q_N, q_S, q_E, q_W, current_phase_idx, elapsed_fraction]` — all normalised to [0, 1]
- **Action space:** 224 = 4 directions × 56 durations (5 s – 60 s, 1 s steps)
- **Action decode:** `direction = action // 56`, `duration = (action % 56) + 5`
- **Architecture:** `Linear(6 → 128) → ReLU → Linear(128 → 64) → ReLU → Linear(64 → 224)` (~42 k params)
- **Training:** Vanilla-DQN with frozen target network, Huber loss, Adam optimiser, experience replay
- **Reward:** `throughput − 0.05 × total_waiting − 0.001 × duration`
- **Weights:** `models/dqn_signal_optimizer.pt`
- **Runtime wrapper:** TODO
- **Fallback:** Proportional timing (green to busiest lane, duration ∝ density ratio) when weights are absent


---
