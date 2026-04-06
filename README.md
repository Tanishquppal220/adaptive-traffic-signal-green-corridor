# Adaptive Traffic Signal — Green Corridor System

> A web-based demo that takes **4 intersection images** (N / S / E / W) + **1 audio file** per session and runs a full ML pipeline: vehicle detection → emergency classification → siren detection → density prediction → DQN signal optimisation — with fairness policies and predictive control.

![Project Banner](image.png)

---

## Table of Contents

- [Features](#features)
- [ML Model Stack](#ml-model-stack)
- [System Architecture](#system-architecture)
- [Repo Structure](#repo-structure)
- [Getting Started](#getting-started)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Training Notebooks](#training-notebooks)
- [Documentation](#documentation)

---

## Features

- **Vehicle Detection** — YOLOv8s fine-tuned single-class detector counts vehicles per lane (N / S / E / W)
- **Emergency Vehicle Classification** — YOLOv8s classifier detects ambulances / fire trucks with calibrated confidence thresholds
- **Siren Detection** — TFLite model on raw WAV audio for siren / no-siren classification
- **Density Prediction** — XGBoost × 4 (per direction) forecasts 60-second-ahead traffic density from a 100-step sliding window
- **DQN Signal Optimiser** — 224-action Deep Q-Network (4 directions × 56 durations) for adaptive green-time allocation
- **Emergency Green Corridor** — Dual-gated emergency policy: activates only when **both** visual detection AND siren audio confirm an emergency
- **Fairness Policy** — Configurable anti-starvation guard (off / soft / hard modes) prevents lane neglect
- **Predictive Control** — EMA smoothing + XGBoost forecast fusion with surge detection and switch-penalty stability
- **Interactive 5-Step Demo UI** — Step-by-step pipeline orchestrator showing each model's output in the browser
- **Online Learning** — DQN continues refining weights after each real cycle via experience replay

---

## ML Model Stack

### 1. Traffic Detection — `trafficDetection.ipynb`
- **Model:** YOLOv8s fine-tuned, single class `vehicle`
- **Input:** One image per direction (N / S / E / W)
- **Output:** `q_N, q_S, q_E, q_W` — vehicle count per lane (via bounding-box centre quadrant mapping)
- **Weights:** `models/traffic_detection_yolov8s.pt`
- **Runtime wrapper:** `control/traffic_detector.py` → `TrafficDetector`
- **Training config:** 80 epochs, img 736, batch 16, AdamW, freeze=10, early stop patience=15

### 2. Emergency Vehicle Classification — `emergnecy_detection.ipynb`
- **Model:** YOLOv8s fine-tuned on emergency vehicle dataset
- **Input:** Same 4 direction images (run after traffic detection)
- **Output:** `emergency_flag = 1` if target class IDs (3 or 5) detected above calibrated threshold
- **Weights:** `models/emergency_vehicle_cls_yolov8s.pt`
- **Threshold:** Auto-loaded from `models/emergency_classifier_threshold.json` (F1-optimal), fallback default 0.5
- **Runtime wrapper:** `control/emergency_classifier.py` → `EmergencyClassifier`

### 3. Siren Detection
- **Model:** Custom TFLite binary classifier
- **Input:** 1 WAV audio file per upload session
- **Output:** `siren_flag = 1` if siren confidence ≥ threshold (default 0.5)
- **Weights:** `models/siren_detector_v2.tflite`
- **Audio pipeline:** WAV decode → mono conversion → linear resampling to 16 kHz → zero-pad to minimum duration → TFLite inference
- **Runtime wrapper:** `control/siren_detector.py` → `SirenDetector`

### 4. Density Predictor — `traffic-Density.ipynb`
- **Model:** XGBoost Booster × 4 (one per direction N / S / E / W)
- **Input:** 100-step sliding window of past counts (400 floats) + 4 cyclic time features (sin/cos hour + sin/cos day-of-week) = **404-dim** feature vector
- **Output:** Predicted avg vehicle count 60 seconds ahead per direction
- **Weights:** `models/density_predictor_{N,S,E,W}.ubj`
- **Runtime wrapper:** `control/density_predictor.py` → `DensityPredictor`
- **Note:** Falls back to heuristic (latest counts) when history < 100 or models unavailable

### 5. DQN Signal Optimiser — `training/DQN/`
- **Framework:** PyTorch
- **State (6-dim):** `[q_N, q_S, q_E, q_W, current_phase_idx, elapsed_fraction]` — all normalised to [0, 1]
- **Action space:** 224 = 4 directions × 56 durations (5 s – 60 s, 1 s steps)
- **Action decode:** `direction = action // 56`, `duration = (action % 56) + 5`
- **Architecture:** `Linear(6 → 128) → ReLU → Linear(128 → 64) → ReLU → Linear(64 → 224)` (~42 k params)
- **Training:** Double-DQN with frozen target network, Huber loss, Adam optimiser, experience replay
- **Reward:** `throughput − 0.05 × total_waiting − 0.002 × duration`
- **Weights:** `models/dqn_signal_optimizer.pt`
- **Runtime wrapper:** `control/signal_controller.py` → `SignalController`
- **Fallback:** Proportional timing (green to busiest lane, duration ∝ density ratio) when weights are absent

---

## System Architecture

```
                     ┌─────────────────────────────────────────────────┐
                     │              Flask Web GUI                     │
                     │   index.html · upload_demo.html                │
                     │   demo_orchestrator.html (5-step pipeline)     │
                     └──────────────────┬──────────────────────────────┘
                                        │  HTTP / JSON
                     ┌──────────────────▼──────────────────────────────┐
                     │            gui/routes.py                        │
                     │   /api/step1..5  ·  /api/run_cycle              │
                     │   /api/next_cycle  ·  /api/status               │
                     └──────────────────┬──────────────────────────────┘
                                        │
                     ┌──────────────────▼──────────────────────────────┐
                     │         ModelController (orchestrator)           │
                     │         control/model_controller.py              │
                     └───┬────┬─────┬─────┬──────┬────────────────────┘
                         │    │     │     │      │
              ┌──────────▼┐ ┌▼─────▼┐ ┌──▼──┐ ┌─▼──────────┐ ┌────────────┐
              │ Traffic    │ │Emerg. │ │Siren│ │  Density   │ │   Signal   │
              │ Detector   │ │Classif│ │Detec│ │  Predictor │ │ Controller │
              │ (YOLOv8s)  │ │(YOLO) │ │(TFL)│ │ (XGBoost)  │ │   (DQN)    │
              └────────────┘ └───────┘ └─────┘ └────────────┘ └────────────┘
```

**Emergency gating policy:** Emergency corridor activates **only** when `emergency_visual AND siren_detected` — prevents false positives from visual-only detection.

---

## Repo Structure

```
adaptive-traffic-signal-green-corridor/
├── main.py                                # Flask entry point
├── config.py                              # Centralised config (all constants & hyperparams)
├── pyproject.toml                         # Project metadata & dependencies (uv / hatch)
│
├── control/                               # Runtime inference modules
│   ├── model_controller.py                # End-to-end pipeline orchestrator
│   ├── traffic_detector.py                # YOLOv8s → per-lane vehicle counts
│   ├── emergency_classifier.py            # YOLOv8s → emergency flag + confidence
│   ├── siren_detector.py                  # TFLite → siren flag from audio
│   ├── density_predictor.py               # XGBoost × 4 → 60 s-ahead density forecast
│   ├── signal_controller.py               # DQN agent wrapper + proportional fallback
│   ├── replay_buffer.py                   # Experience replay for online DQN learning
│   ├── schema.py                          # Lane/direction conversion helpers
│   ├── test_controller.py                 # Unit tests for ModelController
│   ├── test_emergency_classifier_policy.py
│   └── test_runtime_contracts.py
│
├── training/                              # Model training code & notebooks
│   ├── DQN/
│   │   ├── dqn_agent.py                   # QNetwork + DQNAgent (Double-DQN, 224-action)
│   │   ├── environment.py                 # TrafficEnv simulation (Poisson arrivals)
│   │   ├── replay_buffer.py               # Replay buffer for training
│   │   └── train.py                       # Training loop + target network sync
│   ├── trafficDetection.ipynb             # YOLOv8s vehicle detection training
│   ├── emergnecy_detection.ipynb          # YOLOv8s emergency vehicle training
│   └── traffic-Density.ipynb             # XGBoost density predictor training
│
├── models/                                # Trained model weights
│   ├── traffic_detection_yolov8s.pt       # Vehicle detection (22 MB)
│   ├── emergency_vehicle_cls_yolov8s.pt   # Emergency classifier (6 MB)
│   ├── siren_detector_v2.tflite           # Siren audio classifier (5 MB)
│   ├── density_predictor_{N,S,E,W}.ubj    # XGBoost per-direction (1.3 MB each)
│   └── dqn_signal_optimizer.pt            # DQN signal controller (390 KB)
│
├── gui/                                   # Flask web interface
│   ├── __init__.py                        # create_app() factory
│   ├── routes.py                          # All API endpoints + page routes
│   ├── templates/
│   │   ├── index.html                     # Landing page
│   │   ├── upload_demo.html               # 4-image + audio upload form
│   │   └── demo_orchestrator.html         # Interactive 5-step pipeline demo
│   └── static/
│       ├── css/style.css                  # UI styles
│       └── js/app.js                      # Client-side simulation + visualisation
│
├── data/
│   └── README.md                          # Data directory docs
│
├── docs/                                  # Project documentation
│   ├── PROJECT_NOTES.md                   # Detailed project notes
│   ├── Emergency_model_integration.md     # Emergency model integration guide
│   ├── density-predictor-explanation.md   # Density predictor deep-dive
│   ├── demo_logic.md                      # Demo flow documentation
│   └── project-blueprin.md                # Full project blueprint
│
├── plan/
│   └── feature-emergency-classifier-1.md  # Feature plan: emergency classifier
│
├── image.png                              # Project banner image
├── .python-version                        # Python 3.13
└── .gitignore
```

---

## Getting Started

### Prerequisites

- **Python 3.13+**
- **[uv](https://docs.astral.sh/uv/)** package manager (recommended)

### Installation

```bash
# Clone the repo
git clone https://github.com/Tanishquppal220/adaptive-traffic-signal-green-corridor.git
cd adaptive-traffic-signal-green-corridor

# Install dependencies with uv
uv sync
```

### Running the App

```bash
# Default (localhost only, debug off)
uv run python main.py

# Local network demo with debug mode
FLASK_HOST=0.0.0.0 FLASK_DEBUG=1 uv run python main.py
```

The server starts on `http://127.0.0.1:5000` by default.

### Pages

| URL | Description |
|---|---|
| `/` | Landing page |
| `/demo/upload` | Upload 4 images + 1 audio → full pipeline result |
| `/demo/orchestrator` | Interactive 5-step pipeline demo |
| `/api/status` | JSON status of all loaded models |

---

## API Reference

### Pipeline Demo (5-Step Orchestrator)

| Step | Endpoint | Method | Input | Output |
|---|---|---|---|---|
| 1 | `/api/step1/detect-traffic` | POST | 4 images (`laneN`, `laneS`, `laneE`, `laneW`) | Vehicle counts per lane |
| 2 | `/api/step2/detect-emergency` | POST | (uses cached frames from step 1) | Emergency detection result |
| 3 | `/api/step3/detect-siren` | POST | 1 audio file (`sirenAudio`) | Siren detection result |
| 4 | `/api/step4/predict-density` | POST | (uses cached counts from step 1) | XGBoost density predictions |
| 5 | `/api/step5/optimize-signal` | POST | (uses all cached results) | DQN signal decision |

### Full Cycle Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/run_cycle` | POST | Full pipeline: 4 images + audio → detection + decision + simulation payload |
| `/api/next_cycle` | POST | Iterative re-decision from updated lane counts (JSON body) |
| `/api/status` | GET | Health/status of all model subsystems |

---

## Configuration

All configuration lives in `config.py`. Key settings:

### Signal Timing
| Parameter | Default | Description |
|---|---|---|
| `MIN_GREEN` | 5 s | Shortest permissible green phase |
| `MAX_GREEN` | 60 s | Longest permissible green phase |
| `YELLOW_DURATION` | 3 s | Fixed yellow clearance phase |
| `CYCLE_TIMEOUT` | 120 s | Max total cycle before forced rotation |

### DQN Hyperparameters
| Parameter | Default | Description |
|---|---|---|
| `STATE_SIZE` | 6 | State vector dimension |
| `ACTION_SIZE` | 224 | 4 directions × 56 duration steps |
| `GAMMA` | 0.95 | Discount factor |
| `EPSILON_START / END` | 1.0 / 0.05 | ε-greedy exploration bounds |
| `TARGET_UPDATE_FREQ` | 500 | Steps between target network sync |
| `DQN_TRAINING_BATCH_SIZE` | 64 | Training mini-batch size |

### Emergency Policy
| Parameter | Default | Description |
|---|---|---|
| `EMERGENCY_CONFIDENCE_THRESHOLD` | 0.5 | Min confidence for emergency flag |
| `EMERGENCY_TARGET_CLASS_IDS` | (3, 5) | YOLO class IDs for emergency vehicles |
| `EMERGENCY_DURATION_CAP_SEC` | 30 | Max green during emergency corridor |
| `SIREN_CONFIDENCE_THRESHOLD` | 0.5 | Min confidence for siren detection |

### Fairness Policy
| Parameter | Default | Description |
|---|---|---|
| `FAIRNESS_DEFAULT_MODE` | `soft` | `off` / `soft` / `hard` anti-starvation |
| `FAIRNESS_WAIT_THRESHOLD_SEC` | 30.0 | Wait time before fairness triggers |
| `FAIRNESS_MISSED_TURNS_THRESHOLD` | 3 | Missed turns before hard override |

### Predictive Control
| Parameter | Default | Description |
|---|---|---|
| `PREDICTIVE_CONTROL_ENABLED` | `True` | Enable EMA + forecast fusion |
| `PREDICTIVE_ALPHA_CURRENT` | 0.75 | Weight for current observation |
| `PREDICTIVE_BETA_FORECAST` | 0.30 | Weight for XGBoost forecast delta |
| `PREDICTIVE_SURGE_THRESHOLD` | 4.0 | Forecast surge detection threshold |

### Environment Variables
| Variable | Default | Description |
|---|---|---|
| `FLASK_HOST` | `127.0.0.1` | Bind address |
| `FLASK_PORT` | `5000` | Server port |
| `FLASK_DEBUG` | `false` | Debug mode (enables auto-reload) |

---

## RL Action Space — 224 Q-Values

| Dimension | Value |
|---|---|
| Directions | 4 (N, S, E, W) |
| Duration steps | 56 (5 s → 60 s in 1 s increments) |
| Total Q-values | **224** (4 × 56) |
| Action decode | `direction = action // 56`, `duration = (action % 56) + 5` |
| State vector | 6-dim: `[q_N, q_S, q_E, q_W, phase_idx, elapsed_frac]` (normalised [0, 1]) |

---

## Training Notebooks

| Notebook | Location | Description |
|---|---|---|
| `trafficDetection.ipynb` | `training/` | YOLOv8s vehicle detection fine-tuning |
| `emergnecy_detection.ipynb` | `training/` | YOLOv8s emergency vehicle classifier |
| `traffic-Density.ipynb` | `training/` | XGBoost density predictor (4 models) |
| `train.py` | `training/DQN/` | DQN training loop (CLI script) |

---

## Dependencies

Managed via `pyproject.toml` with `uv`:

| Package | Purpose |
|---|---|
| `flask` ≥ 3.0 | Web server |
| `flask-socketio` ≥ 5.6 | WebSocket support |
| `ultralytics` ≥ 8.3 | YOLOv8 inference |
| `torch` ≥ 2.2 | DQN neural network |
| `xgboost` ≥ 3.2 | Density prediction |
| `opencv-python-headless` ≥ 4.10 | Image processing |
| `numpy` ≥ 2.0 | Numerical ops |
| `scikit-learn` ≥ 1.5 | Model utilities |
| `stable-baselines3` ≥ 2.3 | RL utilities |
| `gymnasium` ≥ 0.29 | RL env interface |
| `eventlet` ≥ 0.40 | Async server |
| `pillow` ≥ 9.0 | Image handling |
| `matplotlib` ≥ 3.8 | Training visualisation |

---

## Documentation

Detailed documentation in `docs/`:

- **[PROJECT_NOTES.md](docs/PROJECT_NOTES.md)** — Detailed project notes and decisions
- **[Emergency_model_integration.md](docs/Emergency_model_integration.md)** — Emergency model integration guide
- **[density-predictor-explanation.md](docs/density-predictor-explanation.md)** — Density predictor deep-dive
- **[demo_logic.md](docs/demo_logic.md)** — Demo flow documentation
- **[project-blueprin.md](docs/project-blueprin.md)** — Full project blueprint

---

## License

This project is part of a university coursework submission.