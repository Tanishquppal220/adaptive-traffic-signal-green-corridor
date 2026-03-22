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
