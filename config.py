"""Centralized configuration for the adaptive traffic signal system.

All subsystems (detection, prediction, optimization, audio, ambulance)
should import shared constants from here instead of duplicating values.
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

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
# ── Signal Timing ─────────────────────────────────────────────────────────────
MIN_GREEN = 5  # seconds – shortest permissible green phase
MAX_GREEN = 60  # seconds – longest permissible green phase
YELLOW_DURATION = 3  # seconds – fixed yellow/clearance phase
CYCLE_TIMEOUT = 120  # seconds – max total cycle before forced rotation

# ── Shared Schemas ───────────────────────────────────────────────────────────
DIRECTIONS: tuple[str, ...] = ("N", "S", "E", "W")
LANE_KEYS: tuple[str, ...] = ("laneN", "laneS", "laneE", "laneW")
LANE_TO_DIRECTION: dict[str, str] = {
    "laneN": "N",
    "laneS": "S",
    "laneE": "E",
    "laneW": "W",
}
DIRECTION_TO_LANE: dict[str, str] = {
    v: k for k, v in LANE_TO_DIRECTION.items()}

# ── DQN Action/State Contracts ───────────────────────────────────────────────
N_DIRECTIONS = len(DIRECTIONS)
N_DURATIONS = MAX_GREEN - MIN_GREEN + 1
ACTION_SIZE = N_DIRECTIONS * N_DURATIONS
STATE_SIZE = 6
MAX_VEHICLES_NORM = 30.0

# Backward-compatible action duration options (legacy policy path)
GREEN_DURATIONS: list[int] = [10, 20, 30, 40, 50, 60]

# ── DQN Hyperparameters ───────────────────────────────────────────────────────
DQN_TRAINING_LEARNING_RATE = 1e-3
GAMMA = 0.95  # discount factor
EPSILON_START = 1.0  # full exploration at the start
EPSILON_END = 0.05  # minimum exploration floor
EPSILON_DECAY = 0.9995  # multiplicative decay per step
DQN_TRAINING_BUFFER_SIZE = 50_000
DQN_TRAINING_BATCH_SIZE = 64
TARGET_UPDATE_FREQ = 500  # steps between hard target-network sync

# ── DQN Online / Runtime Hyperparameters ─────────────────────────────────────
DQN_ONLINE_LEARNING_RATE = 1e-4
DQN_ONLINE_BUFFER_SIZE = 5_000
DQN_ONLINE_BATCH_SIZE = 32
DQN_ONLINE_SAVE_EVERY = 100

# Backward-compatible aliases for existing modules
LEARNING_RATE = DQN_TRAINING_LEARNING_RATE
REPLAY_BUFFER_SIZE = DQN_TRAINING_BUFFER_SIZE
BATCH_SIZE = DQN_TRAINING_BATCH_SIZE

# ── Model Persistence ─────────────────────────────────────────────────────────
DQN_WEIGHTS_PATH = MODELS_DIR / "dqn_signal_optimizer.pt"

# ── Density Predictor Runtime ─────────────────────────────────────────────────
DENSITY_HISTORY_WINDOW = 100
DENSITY_PREDICTION_HORIZON_SEC = 60
DENSITY_MAX_CLIP = 50.0

# ── Emergency Model Runtime ───────────────────────────────────────────────────
EMERGENCY_CONFIDENCE_THRESHOLD = 0.5
EMERGENCY_LABEL_KEYWORDS: tuple[str, ...] = ("ambulance", "fire", "police")

# Emergency corridor timing policy
# Applied duration during emergency: min(dqn_duration + buffer, cap)
EMERGENCY_DURATION_BUFFER_SEC = 6
EMERGENCY_DURATION_CAP_SEC = 30

# If emergency lane queue is already low, avoid wasting long green windows.
EMERGENCY_LOW_QUEUE_THRESHOLD = 4
EMERGENCY_LOW_QUEUE_MAX_SEC = 12

# Fairness / anti-starvation policy
FAIRNESS_DEFAULT_MODE: str = "soft"  # off | soft | hard
FAIRNESS_WAIT_THRESHOLD_SEC: float = 30.0
FAIRNESS_MISSED_TURNS_THRESHOLD: int = 3
FAIRNESS_SOFT_WAIT_WEIGHT: float = 0.35
FAIRNESS_SOFT_MISSED_WEIGHT: float = 0.45
FAIRNESS_SOFT_OVERRIDE_MARGIN: float = 2.0

# Synthetic runtime tuning
LOW_TRAFFIC_QUEUE_THRESHOLD: int = 24
LOW_TRAFFIC_MIN_GREEN_FLOOR: int = 7
LOW_TRAFFIC_PER_VEHICLE_BONUS: int = 1
LOW_TRAFFIC_MAX_GREEN_FLOOR: int = 12
LOW_TRAFFIC_PROFILES: tuple[str, ...] = ("night",)
