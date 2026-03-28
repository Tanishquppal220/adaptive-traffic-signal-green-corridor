"""Centralized configuration for the adaptive traffic signal system.

All subsystems (detection, prediction, optimization, audio, ambulance)
should import shared constants from here instead of duplicating values.
"""

import pathlib

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).parent.resolve()
MODELS_DIR = PROJECT_ROOT / "models"

TRAFFIC_DETECTION_MODEL_PATH = MODELS_DIR / "traffic_detection_yolov8s.pt"
EMERGENCY_VEHICLE_MODEL_PATH = MODELS_DIR / "emergency_vehicle_cls_yolov8s.pt"

# ── Density prediction models (trained per direction) ──────────────────────────
DENSITY_PREDICTOR_MODELS = {
    "N": MODELS_DIR / "density_predictor_N.ubj",
    "S": MODELS_DIR / "density_predictor_S.ubj",
    "E": MODELS_DIR / "density_predictor_E.ubj",
    "W": MODELS_DIR / "density_predictor_W.ubj",
}

# ── Detection settings ───────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD: float = 0.5
INFERENCE_INTERVAL: float = 3.0  # seconds between YOLO inferences on live feed

# ── Camera ───────────────────────────────────────────────────────────────────
CAMERA_INDEX: int = 0
CAMERA_BUFFER_SIZE: int = 1

# ── Flask / GUI ──────────────────────────────────────────────────────────────
FLASK_HOST: str = "0.0.0.0"
FLASK_PORT: int = 5000
FLASK_DEBUG: bool = True

# ── RL Signal Control ────────────────────────────────────────────────────────
# Green duration options (seconds) for the RL agent's action space
GREEN_DURATIONS: list[int] = [10, 20, 30, 40, 50, 60]

# ── Directories (add if not already present) ──────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ── Signal Timing ─────────────────────────────────────────────────────────────
MIN_GREEN = 5    # seconds – shortest permissible green phase
MAX_GREEN = 60   # seconds – longest permissible green phase
YELLOW_DURATION = 3    # seconds – fixed yellow/clearance phase
CYCLE_TIMEOUT = 120  # seconds – max total cycle before forced rotation

# ── DQN Hyperparameters ───────────────────────────────────────────────────────
LEARNING_RATE = 1e-3
GAMMA = 0.95       # discount factor
EPSILON_START = 1.0        # full exploration at the start
EPSILON_END = 0.05       # minimum exploration floor
EPSILON_DECAY = 0.9995     # multiplicative decay per step
REPLAY_BUFFER_SIZE = 50_000
BATCH_SIZE = 64
TARGET_UPDATE_FREQ = 500        # steps between hard target-network sync

# ── Model Persistence ─────────────────────────────────────────────────────────
DQN_WEIGHTS_PATH = MODELS_DIR / "dqn_signal_optimizer.pt"
