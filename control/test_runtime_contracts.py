from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg  # noqa: E402
from control.emergency_classifier import EmergencyClassifier  # noqa: E402
from control.model_controller import ModelController  # noqa: E402
from gui.routes import (  # noqa: E402
    _apply_fixed_round_robin_override,
    _apply_low_traffic_duration_policy,
    _apply_non_repeat_scheduler,
)
from training.DQN.environment import decode_action, encode_action  # noqa: E402


def _assert_action_matches_decision(decision: dict, label: str) -> None:
    action = decision.get("action")
    if not isinstance(action, int):
        raise AssertionError(
            f"{label}: action must be int, got {type(action).__name__}")

    decoded_direction_idx, decoded_duration = decode_action(action)
    decoded_direction = cfg.DIRECTIONS[decoded_direction_idx]

    expected_direction = str(decision.get("direction", "N"))
    expected_duration = int(decision.get("duration", 0))

    if decoded_direction != expected_direction:
        raise AssertionError(
            f"{label}: decoded direction {decoded_direction} != expected {expected_direction}"
        )
    if decoded_duration != expected_duration:
        raise AssertionError(
            f"{label}: decoded duration {decoded_duration} != expected {expected_duration}"
        )


def _build_decision(direction: str, duration: int, mode: str = "test") -> dict:
    lane_values = {lane: 0 for lane in cfg.LANE_KEYS}
    lane_values[cfg.DIRECTION_TO_LANE[direction]] = duration
    action = encode_action(cfg.DIRECTIONS.index(direction), duration)
    return {
        **lane_values,
        "direction": direction,
        "duration": duration,
        "action": action,
        "mode": mode,
        "fairness": {
            "mode": "soft",
            "applied": False,
            "reason": "baseline_retained",
            "selected_lane": cfg.DIRECTION_TO_LANE[direction],
            "selected_duration": duration,
        },
        "baseline_decision": {
            "direction": direction,
            "duration": duration,
            "action": action,
            "mode": mode,
        },
    }


def _run_model_controller_contract_checks() -> None:
    controller = ModelController()

    emergency_result = controller.decide_from_lane_counts(
        lane_counts={"laneN": 8, "laneS": 3, "laneE": 0, "laneW": 0},
        emergency_override={
            "detected": True,
            "label": "ambulance-test",
            "confidence": 1.0,
            "direction": "S",
            "predictions": [],
            "mode": "contract-test",
        },
        fairness_mode="off",
    )
    _assert_action_matches_decision(emergency_result, "emergency override")

    for lane in cfg.LANE_KEYS:
        controller._fairness_state[lane]["wait_seconds"] = 0.0
        controller._fairness_state[lane]["missed_turns"] = 0
    controller._fairness_state["laneS"]["wait_seconds"] = float(
        cfg.FAIRNESS_WAIT_THRESHOLD_SEC + 5)
    controller._fairness_state["laneS"]["missed_turns"] = int(
        cfg.FAIRNESS_MISSED_TURNS_THRESHOLD + 1
    )

    fairness_hard_decision, fairness_hard_info = controller._apply_fairness_policy(
        lane_counts={"laneN": 1, "laneS": 9, "laneE": 0, "laneW": 0},
        baseline_decision=_build_decision(
            direction="N", duration=12, mode="baseline-hard"),
        fairness_mode="hard",
    )
    if not fairness_hard_info.get("applied"):
        raise AssertionError("fairness hard: expected override to be applied")
    _assert_action_matches_decision(
        fairness_hard_decision, "fairness hard override")

    for lane in cfg.LANE_KEYS:
        controller._fairness_state[lane]["wait_seconds"] = 0.0
        controller._fairness_state[lane]["missed_turns"] = 0

    fairness_soft_decision, fairness_soft_info = controller._apply_fairness_policy(
        lane_counts={"laneN": 1, "laneS": 0, "laneE": 15, "laneW": 0},
        baseline_decision=_build_decision(
            direction="N", duration=10, mode="baseline-soft"),
        fairness_mode="soft",
    )
    if not fairness_soft_info.get("applied"):
        raise AssertionError("fairness soft: expected override to be applied")
    _assert_action_matches_decision(
        fairness_soft_decision, "fairness soft override")


def _run_route_override_contract_checks() -> None:
    low_traffic_result, low_traffic_adjusted = _apply_low_traffic_duration_policy(
        result=_build_decision(direction="N", duration=5, mode="low-traffic"),
        lane_counts={"laneN": 5, "laneS": 0, "laneE": 0, "laneW": 0},
        profile_name="night",
    )
    if not low_traffic_adjusted:
        raise AssertionError(
            "low traffic policy: expected duration floor adjustment")
    _assert_action_matches_decision(
        low_traffic_result, "low traffic duration floor")

    fixed_rr_result, _ = _apply_fixed_round_robin_override(
        result=_build_decision(direction="N", duration=10, mode="fixed-rr"),
        lane_counts={"laneN": 2, "laneS": 3, "laneE": 1, "laneW": 0},
        last_served_lane="laneN",
    )
    _assert_action_matches_decision(fixed_rr_result, "fixed round robin")

    non_repeat_result, non_repeat_meta = _apply_non_repeat_scheduler(
        result=_build_decision(direction="N", duration=10, mode="scheduler"),
        lane_counts={"laneN": 5, "laneS": 6, "laneE": 2, "laneW": 1},
        last_served_lane="laneN",
    )
    if not non_repeat_meta.get("selected_by_scheduler"):
        raise AssertionError(
            "non-repeat scheduler: expected lane rotation to be applied")
    _assert_action_matches_decision(non_repeat_result, "non-repeat scheduler")


def _run_emergency_label_policy_contract_checks() -> None:
    classifier = EmergencyClassifier(
        model_path=cfg.MODELS_DIR / "__missing_policy_test_model__.pt"
    )

    if not classifier._is_target_class(3):
        raise AssertionError(
            "emergency class-id policy: expected class id 3 to trigger")

    if not classifier._is_target_class(5):
        raise AssertionError(
            "emergency class-id policy: expected class id 5 to trigger")

    if classifier._is_target_class(1):
        raise AssertionError(
            "emergency class-id policy: class id 1 must not trigger"
        )

    for label in ("fire", "fire-truck", "police", "non_emergency", "1"):
        if classifier._is_emergency(label):
            raise AssertionError(
                f"emergency label policy: label {label!r} must not trigger"
            )


def run_tests() -> None:
    _run_model_controller_contract_checks()
    _run_route_override_contract_checks()
    _run_emergency_label_policy_contract_checks()
    print("Runtime contract checks passed.")


if __name__ == "__main__":
    run_tests()
