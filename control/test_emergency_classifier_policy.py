from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg  # noqa: E402
from control.emergency_classifier import EmergencyClassifier  # noqa: E402


class _FakeResult:
    def __init__(self, boxes: "_FakeBoxes", names: dict[int, str]) -> None:
        self.boxes = boxes
        self.names = names


class _FakeListValue:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return self._values


class _FakeScalar:
    def __init__(self, value: float) -> None:
        self._value = float(value)

    def item(self) -> float:
        return self._value


class _FakeBoxes:
    def __init__(self, cls_ids: list[int], confidences: list[float]) -> None:
        self.cls = [_FakeScalar(v) for v in cls_ids]
        self.conf = [_FakeScalar(v) for v in confidences]
        self.xyxy = [
            _FakeListValue([10.0 + i, 20.0 + i, 50.0 + i, 80.0 + i])
            for i in range(len(cls_ids))
        ]

    def __len__(self) -> int:
        return len(self.cls)


class _FakeModel:
    def __init__(
        self,
        cls_ids: list[int],
        confidences: list[float],
        names: dict[int, str] | None = None,
    ) -> None:
        self._cls_ids = cls_ids
        self._confidences = confidences
        self._names = names or {}

    def predict(
        self,
        _crop: np.ndarray,
        conf: float | None = None,
        verbose: bool = False,
    ) -> list[_FakeResult]:
        _ = conf
        _ = verbose
        boxes = _FakeBoxes(self._cls_ids, self._confidences)
        return [_FakeResult(boxes=boxes, names=self._names)]


def _build_classifier(
    *,
    class_id: int,
    confidence: float,
    threshold: float = cfg.EMERGENCY_CONFIDENCE_THRESHOLD,
) -> EmergencyClassifier:
    classifier = EmergencyClassifier(
        model_path=cfg.MODELS_DIR / "__missing_emergency_classifier_policy_model__.pt",
        confidence_threshold=threshold,
    )
    classifier._model = _FakeModel(
        cls_ids=[class_id],
        confidences=[confidence],
        names={class_id: str(class_id)},
    )
    classifier._error = None
    return classifier


def _assert_ambulance_triggers() -> None:
    classifier = _build_classifier(
        class_id=3, confidence=0.95, threshold=0.5)
    frame = np.zeros((96, 96, 3), dtype=np.uint8)
    result = classifier.classify(frame)

    if not result.get("detected"):
        raise AssertionError(
            "ambulance label should trigger emergency detection")


def _assert_non_ambulance_does_not_trigger() -> None:
    classifier = _build_classifier(
        class_id=1, confidence=0.99, threshold=0.5)
    frame = np.zeros((96, 96, 3), dtype=np.uint8)
    result = classifier.classify(frame)

    if result.get("detected"):
        raise AssertionError(
            "non-ambulance label must not trigger emergency detection")

    predictions = result.get("predictions", [])
    if predictions and predictions[0].get("is_emergency"):
        raise AssertionError(
            "non-ambulance prediction must remain non-emergency")


def _assert_threshold_blocks_low_confidence() -> None:
    classifier = _build_classifier(
        class_id=5, confidence=0.30, threshold=0.5)
    frame = np.zeros((96, 96, 3), dtype=np.uint8)
    result = classifier.classify(frame)

    if result.get("detected"):
        raise AssertionError(
            "ambulance below threshold must not trigger emergency detection")


def run_tests() -> None:
    _assert_ambulance_triggers()
    _assert_non_ambulance_does_not_trigger()
    _assert_threshold_blocks_low_confidence()
    print("Emergency classifier policy tests passed.")


if __name__ == "__main__":
    run_tests()
