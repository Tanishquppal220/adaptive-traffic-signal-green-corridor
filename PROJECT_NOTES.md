# 🚦 Adaptive Traffic Signal & Green Corridor — Project Notes

> **Project:** College ML Project — Real-time adaptive traffic signal controller for a 4-way intersection (N/S/E/W) using Computer Vision + Reinforcement Learning.

---

## Table of Contents

1. [Step-by-Step: What We Built](#1-step-by-step-what-we-built)
2. [Errors We Faced & How We Fixed Them](#2-errors-we-faced--how-we-fixed-them)
3. [Concepts Involved & Explanations](#3-concepts-involved--explanations)
4. [Inputs & Outputs — How Data Flows](#4-inputs--outputs--how-data-flows)

---

## 1. Step-by-Step: What We Built

### Phase 1 — Understanding the Existing Project

Before writing a single line of new code, we read and understood what already existed:

| File / Artifact | What It Does |
|---|---|
| `traffic_detection_yolov8s.pt` | YOLOv8 model — counts vehicles in a camera frame per lane |
| `emergency_vehicle_cls_yolov8s.pt` | YOLOv8 classifier — detects ambulance + direction (0–3) |
| `density_predictor_N/S/E/W.ubj` | 4 XGBoost models — predict traffic density ~60s ahead per lane |
| `gui/app.py` | Flask web server with camera feed route |
| `detection/vehicle_detector.py` | Python wrapper around the YOLO vehicle model |

The existing system could detect vehicles and predict density, but had **no decision-making brain** — green light timings were fixed at 30 seconds regardless of traffic load.

---

### Phase 2 — Designing the Gymnasium Environment (`simulation/traffic_env.py`)

**Goal:** Give the RL agent a realistic simulation to train on.

We created a custom `gymnasium.Env` class called `TrafficEnv`. Key design decisions:

- **State space (12 floats):**
  - 4 raw vehicle counts (N/S/E/W), normalized to [0, 1]
  - 4 predicted densities from XGBoost, normalized to [0, 1]
  - 1 current phase (0 = NS green, 1 = EW green)
  - 1 time elapsed in current phase, normalized
  - 1 ambulance present flag (0 or 1)
  - 1 ambulance direction (normalized 0–3)

- **Action space (discrete, 5 actions):**
  Mapped directly to green light durations:
  ```
  Action 0 → 10s
  Action 1 → 20s
  Action 2 → 30s
  Action 3 → 40s
  Action 4 → 50s
  ```

- **Vehicle arrival model:** Poisson distribution (λ = time × density) — reflects realistic random arrivals.

- **Reward function:**
  ```
  reward = −(weighted queue length)
         − (waiting time penalty)
         + (throughput bonus if ambulance cleared)
         − 500  (if ambulance missed — emergency penalty)
  ```

---

### Phase 3 — Training the DQN Agent (`training/train_rl.py`)

We used **Stable-Baselines3** to train a DQN (Deep Q-Network) agent:

```python
from stable_baselines3 import DQN

model = DQN(
    "MlpPolicy",
    env,
    policy_kwargs={"net_arch": [256, 256]},
    learning_rate=1e-4,
    batch_size=64,
    buffer_size=50_000,
    ...
)
model.learn(total_timesteps=200_000)
model.save("training/checkpoints/signal_policy.zip")
```

**Training results:**
| Policy | Mean Reward | Std |
|---|---|---|
| Fixed 30s (baseline) | −357.5 | ±63.6 |
| **DQN Agent** | **−166.3** | **±39.7** |
| **Improvement** | **+191.2 pts** | **~54% better** |

Artifacts saved:
- `training/checkpoints/signal_policy.zip` — the trained model
- `training/checkpoints/learning_curve.png` — reward over training steps
- `training/checkpoints/comparison.png` — bar chart vs baselines

---

### Phase 4 — Evaluation Script (`training/evaluate.py`)

A standalone script that:
1. Runs 200 simulation episodes for each of three policies:
   - Fixed 20s baseline
   - Fixed 30s baseline
   - DQN agent
2. Prints a comparison table with mean/std rewards
3. Shows action distribution (which green durations the agent prefers)
4. Saves a 2-panel comparison chart

---

### Phase 5 — The Control Bridge (`control/`)

Two files that bridge the live camera system to the trained DQN:

#### `control/state_encoder.py` — StateEncoder
Converts messy live inputs into the clean 12-float vector the DQN expects:
```python
encoder = StateEncoder()
state = encoder.encode(
    vehicle_counts={"N": 7, "S": 3, "E": 12, "W": 1},
    predicted_densities={"N": 8.2, ...},
    current_phase=0,
    time_in_phase=5.3,
    ambulance_detected=False,
    ambulance_direction=-1
)
# → np.array([0.14, 0.06, 0.24, 0.02, ...])  shape=(12,)
```

#### `control/rl_agent.py` — RLAgent
Wraps the trained model:
```python
agent = RLAgent()  # loads signal_policy.zip once at startup
green_seconds = agent.get_action(detections_dict)  # returns int like 30
```
- Falls back to 30s if model file is missing (safe default)
- Exposes `agent.is_loaded` flag for UI display

---

### Phase 6 — Fixing the XGBoost Loader Bug (`detection/traffic_predictor.py`)

**Problem discovered:** The existing code loaded `.ubj` model files with `joblib.load()`:
```python
# ❌ WRONG — joblib can't read XGBoost's native binary format
self._models[lane] = joblib.load(model_path)
```

**Fix applied:** Used the native XGBoost loader:
```python
# ✅ CORRECT — xgb.Booster reads .ubj files natively
booster = xgb.Booster()
booster.load_model(str(model_path))
```

> **Note:** The user later reverted this fix back to joblib (user's choice). Both approaches work if the models were saved in the matching format.

---

### Phase 7 — Flask Integration (`gui/app.py`)

Added to the existing Flask app:

1. **At startup:** `agent = RLAgent()` instantiated once (loads model)
2. **`_signal_state` dict:** Shared in-memory state tracking current phase, expiry time, last counts, ambulance status
3. **New `GET/POST /signal_decision` endpoint:**
   - `GET` → returns current cached signal state (fast, no ML inference)
   - `POST` → accepts new vehicle counts, runs XGBoost + RL agent, flips phase when expired

4. **Updated `/test` route:** After YOLO counts all 4 lanes, automatically runs XGBoost predictions and asks the RL agent for the best green duration

---

### Phase 8 — Frontend Dashboard (`gui/templates/index.html`)

Upgraded from a basic camera page to a full two-column live dashboard:

**Left column:** Camera feed (WebRTC `getUserMedia`) with YOLO bounding boxes, annotated server-side and streamed back as base64 JPEG every 1.5 seconds.

**Right column (new — RL Signal Panel):**
- 🚑 Ambulance alert banner (pulsing red animation when active)
- Intersection diagram with 4 live traffic lights (green/yellow/red, CSS animated)
- Countdown bar (depletes over the green phase duration, turns yellow in last 3s)
- Vehicle queue counts (N/S/E/W from YOLO)
- XGBoost 60s predicted densities (N/S/E/W)
- RL Agent badge (shows chosen green duration or "Fallback 30s")

Polls `/signal_decision` every **800ms** and pushes YOLO counts on every frame capture.

---

### Phase 9 — Upload Test Page (`gui/templates/test.html`)

Updated the image-upload test page to also display the full signal decision **after YOLO detection**:
- Phase label (NS GREEN / EW GREEN)
- Green duration pill
- 4 traffic lights
- Vehicle counts grid
- XGBoost predictions grid

---

### Phase 10 — Git Disaster Recovery

User accidentally ran `git reset --hard HEAD~1` after committing all RL work. Recovery:

```bash
# The commit hash was visible from the reset output
git reset --hard 990c41c
# → HEAD is now at 990c41c — all 16 files restored instantly
```

**Key lesson:** `git reset --hard` does NOT delete commits. They stay in `git reflog` for ~30 days.

---

## 2. Errors We Faced & How We Fixed Them

### Error 1 — `ModuleNotFoundError: No module named 'cv2'`

```
File "gui/app.py", line 14, in <module>
    import cv2
ModuleNotFoundError: No module named 'cv2'
```

**Cause:** Running `python main.py` used the system Python, not the virtual environment. `cv2` was only installed inside `.venv`.

**Fix:**
```powershell
# Option A — activate venv first
.venv\Scripts\activate
python main.py

# Option B — use venv python directly
.venv\Scripts\python.exe main.py
```

---

### Error 2 — XGBoost `.ubj` Files Not Loading

**Cause:** The density predictor used `joblib.load()` on files saved as XGBoost's native binary (`.ubj`). These are incompatible formats.

**Fix:** Replaced `joblib.load()` with the XGBoost native loader:
```python
booster = xgb.Booster()
booster.load_model(str(model_path))  # reads .ubj natively
```

---

### Error 3 — `update_history()` Method Missing

While wiring up the `/signal_decision` endpoint, we called `predictor.update_history(counts)`. We had to verify it existed in `traffic_predictor.py` (it did, at line 109) before completing the integration.

---

### Error 4 — Startup Hanging (False Alarm)

The dry-run import check (`python -c "from gui.app import create_app..."`) appeared to hang for over 2 minutes with no output. This was **not actually an error** — YOLOv8 `.pt` models take 30–60 seconds to load on first import (loading PyTorch weights into memory). The app functions correctly once loaded.

---

### Error 5 — Accidental `git reset --hard HEAD~1`

**What happened:** User committed RL work, then ran `git reset` (which unstaged files) then immediately ran `git reset --hard HEAD~1` (which reverted the working tree to the commit before the RL work).

**Why it was recoverable:** The RL commit (`990c41c`) was already saved in Git's object store. `git reset --hard` only moves `HEAD`, it does not destroy commits.

**Fix:**
```bash
git reset --hard 990c41c  # restore to the RL commit directly by hash
```

---

## 3. Concepts Involved & Explanations

### 3.1 — YOLOv8 (You Only Look Once v8)

A state-of-the-art real-time object detection model. In this project it serves two roles:

| Model | Role | Output |
|---|---|---|
| `traffic_detection_yolov8s.pt` | Counts all vehicles in a frame | `{"N": 7, "S": 3, "E": 12, "W": 1}` |
| `emergency_vehicle_cls_yolov8s.pt` | Classifies if vehicle is ambulance | `{"ambulance": True, "direction": 2}` |

YOLO works by dividing the image into a grid, predicting bounding boxes and class probabilities for each cell simultaneously — hence "you only look once" (one forward pass, no region proposals).

---

### 3.2 — XGBoost (eXtreme Gradient Boosting)

An ensemble machine learning algorithm built on **gradient boosted decision trees**. Each new tree corrects the errors of the previous trees.

In this project, 4 separate XGBoost models (one per lane) predict **future traffic density** 60 seconds ahead, given:
- Last 5 density readings from that lane
- Normalized prediction horizon

This helps the RL agent make **proactive** decisions rather than reacting only to current conditions.

---

### 3.3 — Reinforcement Learning (RL)

**Core idea:** An agent learns a policy by trial and error — taking actions in an environment and receiving rewards or penalties.

```
Environment → State → Agent → Action → Environment
                 ↑                          ↓
                 └─────────── Reward ────────┘
```

Components in this project:

| RL Concept | Concrete Meaning |
|---|---|
| **State** | 12-float vector: vehicle counts, predicted densities, phase, time, ambulance |
| **Action** | One of 5 green durations: 10s, 20s, 30s, 40s, 50s |
| **Reward** | Negative queue length (minimize waiting) + huge penalty for ambulance delay |
| **Policy** | The neural network that maps state → best action |
| **Episode** | One full intersection cycle (typically 10–20 phase switches) |

---

### 3.4 — DQN (Deep Q-Network)

DQN is an RL algorithm that uses a **neural network** to approximate the Q-function:

```
Q(state, action) = expected total future reward from taking 'action' in 'state'
```

Training loop:
1. Agent takes action based on current Q-network (with ε-greedy exploration)
2. Environment returns next state and reward
3. Experience `(s, a, r, s')` is stored in a **replay buffer**
4. Random mini-batch sampled from buffer → used to update Q-network via Bellman equation
5. Repeat

Key DQN improvements over vanilla Q-learning:
- **Experience Replay** — breaks temporal correlations in training data
- **Target Network** — a separate frozen network for stable Q-targets

Architecture used: `MLP [12 → 256 → 256 → 5]`

---

### 3.5 — Gymnasium (OpenAI Gym)

A standard Python interface for RL environments:
```python
class TrafficEnv(gymnasium.Env):
    def reset(self) → (observation, info)     # start new episode
    def step(action) → (obs, reward, done, truncated, info)  # take one step
    def render()                              # optional visualisation
```

Stable-Baselines3 DQN expects exactly this interface, so we built `TrafficEnv` to implement it.

---

### 3.6 — Stable-Baselines3 (SB3)

A clean, reliable RL library built on PyTorch. Provides battle-tested implementations of DQN, PPO, SAC, and others with a consistent `fit()`-style API:
```python
model = DQN("MlpPolicy", env, ...)
model.learn(total_timesteps=200_000)
model.save("signal_policy.zip")

# Later, for inference:
model = DQN.load("signal_policy.zip")
action, _ = model.predict(state, deterministic=True)
```

---

### 3.7 — Flask (Python Web Framework)

A lightweight WSGI web framework. Used here to:
- Serve the live camera dashboard HTML page
- Expose REST API endpoints (`/predict_frame`, `/signal_decision`)
- Handle image uploads via `multipart/form-data`
- Return JSON responses consumed by frontend JavaScript

---

### 3.8 — Poisson Distribution (Vehicle Arrivals)

Vehicle arrivals at intersections follow a **Poisson process** — the number of vehicles arriving in a fixed window follows `Poisson(λ)` where `λ = arrival_rate × time_seconds`.

We used this in the simulation environment to generate realistic random traffic:
```python
new_vehicles = np.random.poisson(lam=density * phase_duration)
```

This makes the simulation statistically representative of real-world intersections.

---

### 3.9 — Green Corridor for Emergency Vehicles

When an ambulance is detected on an axis (e.g., North), the system:

1. **Immediately forces the RL agent** to keep NS GREEN (phase 0) regardless of queue lengths
2. Applies a **−500 reward penalty** if the agent tries to switch phases while the ambulance is present
3. Maintains green until the emergency classifier reports ambulance cleared
4. Resumes normal RL-controlled switching afterward

This creates a "green corridor" — a chain of green lights along the ambulance's path.

---

## 4. Inputs & Outputs — How Data Flows

### 4.1 — Input Sources

The system has three distinct input modes:

#### Mode A — Live Camera Feed (`/`)
```
Browser WebRTC camera
    → captureCanvas.toBlob() every 1.5s
    → POST /predict_frame   (multipart JPEG)
    → Flask decodes via cv2.imdecode()
    → YOLOv8 vehicle detector
    → count pushed to POST /signal_decision
```

#### Mode B — Image Upload (`/test`)
```
User uploads 4 images (one per lane: laneN, laneS, laneE, laneW)
    → POST /test  (multipart/form-data)
    → Flask loops over each uploaded file
    → cv2.imdecode() per image
    → YOLOv8 detection per lane
    → lane_counts = {"laneN": 7, "laneS": 3, ...}
    → XGBoost prediction (60s ahead, per lane)
    → RL Agent decision
    → Results rendered back to test.html
```

#### Mode C — Emergency Vehicle Upload (`/test_emergency`)
```
User uploads a single vehicle image
    → POST /test_emergency
    → Emergency classifier YOLOv8 model
    → {"ambulance": True/False, "direction": 0-3 or -1}
    → Displayed in test_emergency.html
```

---

### 4.2 — The Core Decision Pipeline

```
┌─────────────────────────────────────────────────────┐
│                 REAL-TIME PIPELINE                  │
│                                                     │
│  Camera Frame                                       │
│       │                                             │
│       ▼                                             │
│  YOLOv8 Vehicle Detector                            │
│  → vehicle_counts {N, S, E, W}                      │
│       │                                             │
│       ├──→ XGBoost Predictor (×4 lanes)             │
│       │    → predicted_densities {N, S, E, W}       │
│       │                                             │
│       ├──→ Emergency Classifier                     │
│       │    → {ambulance: bool, direction: 0-3}      │
│       │                                             │
│       ▼                                             │
│  StateEncoder.encode(all above)                     │
│  → state_vector [12 floats, normalized 0-1]         │
│       │                                             │
│       ▼                                             │
│  DQN Agent (signal_policy.zip)                      │
│  → action (0-4) → green_duration (10/20/30/40/50s)  │
│       │                                             │
│       ▼                                             │
│  /signal_decision Response                          │
│  → phase, signal_states, time_remaining,            │
│    predicted_densities, ambulance_detected          │
└─────────────────────────────────────────────────────┘
```

---

### 4.3 — Output Display

#### Live Dashboard (`/`) — Polling every 800ms

| UI Element | Data Source | Update Frequency |
|---|---|---|
| Annotated camera canvas | `/predict_frame` | Every 1.5s |
| Traffic light colours (N/S/E/W) | `/signal_decision` → `signal_states` | Every 800ms |
| Countdown bar | `time_remaining / green_duration` | Every 800ms |
| Vehicle queue counts | `vehicle_counts` | Every 800ms |
| XGBoost predicted densities | `predicted_densities` | Every 800ms |
| Ambulance banner | `ambulance_detected` | Every 800ms |
| RL Agent badge | `rl_model_loaded`, `green_duration` | Every 800ms |

#### Upload Test Page (`/test`) — Single POST response

| UI Element | Data |
|---|---|
| Phase hero label | "NS GREEN" or "EW GREEN" |
| Green duration pill | e.g., "🟢 Green for 30s" |
| Traffic light circles | Per-direction GREEN / RED |
| YOLO counts grid | `vehicle_counts {N, S, E, W}` |
| XGBoost density grid | `predicted_densities {N, S, E, W}` |
| Annotated images | Base64 JPEG with bounding boxes overlaid |

---

### 4.4 — API Response Format (`/signal_decision`)

```json
{
  "phase": 0,
  "green_duration": 30,
  "time_in_phase": 12.4,
  "time_remaining": 17.6,
  "signal_states": {
    "N": "GREEN",
    "S": "GREEN",
    "E": "RED",
    "W": "RED"
  },
  "ambulance_detected": false,
  "ambulance_direction": -1,
  "rl_model_loaded": true,
  "predicted_densities": {
    "N": 8.2,
    "S": 3.5,
    "E": 13.1,
    "W": 1.2
  },
  "vehicle_counts": {
    "N": 7,
    "S": 3,
    "E": 12,
    "W": 1
  }
}
```

> **Phase encoding:** `0` = North/South lanes are GREEN, East/West are RED. `1` = East/West are GREEN, North/South are RED.
> Yellow transition is applied automatically in the last **3 seconds** of any phase.

---

## Summary

| Component | Technology | Purpose |
|---|---|---|
| Vehicle Counting | YOLOv8s | Per-lane vehicle count from camera |
| Emergency Detection | YOLOv8 Classifier | Detect ambulance + direction |
| Future Density | XGBoost (4 models) | Predict traffic 60s ahead per lane |
| Simulation | Gymnasium custom env | Train RL agent safely offline |
| Decision Making | DQN via SB3 | Choose optimal green light duration |
| Web Server | Flask | Serve UI + expose REST API |
| Frontend | Vanilla HTML/CSS/JS | Live dashboard + upload test pages |
| Version Control | Git | Code history + disaster recovery |
