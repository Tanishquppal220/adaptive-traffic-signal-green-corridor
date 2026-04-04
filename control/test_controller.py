"""
optimization/test_controller.py
────────────────────────────────
Integration test for the 224-action SignalController.

Tests
─────
1. DQN mode (if weights exist): direction is valid, duration is a whole
   number in [MIN_GREEN, MAX_GREEN], green-lane matches direction label.
2. Proportional fallback: works correctly when no weights file is present.
3. Edge cases: empty intersection, max load, single-lane load.
4. Decode consistency: action → direction/duration roundtrip.

Run
───
    python -m  test_controller
    python -m  test_controller --weights models/dqn_signal_optimizer.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from control.signal_controller import SignalController  # noqa: E402
from training.DQN.environment import (  # noqa: E402
    ACTION_SIZE,
    MAX_GREEN,
    MIN_GREEN,
    decode_action,
    encode_action,
)

try:
    import config as cfg

    LANE_KEYS = cfg.LANE_KEYS
    DIR_LABELS = cfg.DIRECTIONS
except (ImportError, AttributeError):
    LANE_KEYS = ("laneN", "laneS", "laneE", "laneW")
    DIR_LABELS = ("N", "S", "E", "W")


# ── validation helper ──────────────────────────────────────────────────────────

def _validate_action_contract(result: dict) -> list[str]:
    errors: list[str] = []
    action = result["action"]
    if not (0 <= action < ACTION_SIZE):
        errors.append(f"action {action} out of range [0, {ACTION_SIZE-1}]")
        return errors

    d_idx, d_dur = decode_action(action)
    if DIR_LABELS[d_idx] != result["direction"]:
        errors.append(
            f"action {action} decodes to direction {DIR_LABELS[d_idx]} "
            f"but result says {result['direction']}"
        )
    if d_dur != result["duration"]:
        errors.append(
            f"action {action} decodes to duration {d_dur} "
            f"but result says {result['duration']}"
        )
    return errors


def _validate(result: dict, lane_counts: dict[str, int]) -> list[str]:
    """Return a list of error strings (empty = all OK)."""
    errors = []

    # 1. Required keys present
    for key in (*LANE_KEYS, "direction", "duration", "action", "mode"):
        if key not in result:
            errors.append(f"Missing key: '{key}'")

    if errors:
        return errors   # can't proceed without all keys

    # 2. Duration is a whole number in [MIN_GREEN, MAX_GREEN]
    dur = result["duration"]
    if not isinstance(dur, int):
        errors.append(f"duration must be int, got {type(dur).__name__}")
    elif not (MIN_GREEN <= dur <= MAX_GREEN):
        errors.append(f"duration {dur} outside [{MIN_GREEN}, {MAX_GREEN}]")

    # 3. Direction label is valid
    if result["direction"] not in DIR_LABELS:
        errors.append(f"direction '{result['direction']}' not in {DIR_LABELS}")

    # 4. Exactly one lane has non-zero green time and it matches direction label
    active_lanes = [k for k in LANE_KEYS if result[k] > 0]
    if len(active_lanes) != 1:
        errors.append(
            f"Expected exactly 1 active lane, got {len(active_lanes)}: {active_lanes}"
        )
    else:
        active_dir = active_lanes[0].replace("lane", "")   # "laneN" → "N"
        if active_dir != result["direction"]:
            errors.append(
                f"Active lane '{active_dir}' doesn't match direction '{result['direction']}'"
            )

    # 5. The active lane's green time equals the reported duration
    expected_key = f"lane{result['direction']}"
    if result.get(expected_key) != dur:
        errors.append(
            f"lane{result['direction']}={result.get(expected_key)} "
            f"but duration={dur}"
        )

    # 6. action-direction-duration contract is consistent
    errors.extend(_validate_action_contract(result))

    return errors


# ── test suite ─────────────────────────────────────────────────────────────────

def run_tests(weights_path: Path | None = None) -> None:
    print("\n" + "═" * 70)
    print("  SignalController — 224-Action Integration Test")
    print("═" * 70)

    sc = SignalController(
        weights_path=weights_path or ROOT / "models" / "dqn_signal_optimizer.pt"
    )
    print(f"\n  Mode   : {sc.mode}")
    print(f"  Status : {sc.status()}\n")

    test_cases = [
        ("Heavy North",    {"laneN": 15,
         "laneS": 2,  "laneE": 3,  "laneW": 1}),
        ("Heavy South",    {"laneN": 1,
         "laneS": 18, "laneE": 2,  "laneW": 0}),
        ("Heavy East",     {"laneN": 3,
         "laneS": 2,  "laneE": 20, "laneW": 4}),
        ("Heavy West",     {"laneN": 2,
         "laneS": 1,  "laneE": 3,  "laneW": 14}),
        ("Uniform load",   {"laneN": 5,
         "laneS": 5,  "laneE": 5,  "laneW": 5}),
        ("Empty",          {"laneN": 0,
         "laneS": 0,  "laneE": 0,  "laneW": 0}),
        ("Max load",       {"laneN": 30,
         "laneS": 30, "laneE": 30, "laneW": 30}),
        ("API example",    {"laneN": 8,
         "laneS": 2,  "laneE": 5,  "laneW": 1}),
        ("Near-equal N/S", {"laneN": 10,
         "laneS": 9,  "laneE": 1,  "laneW": 1}),
        ("Single lane",    {"laneN": 20,
         "laneS": 0,  "laneE": 0,  "laneW": 0}),
    ]

    passed = 0
    failed = 0

    print(
        f"  {'Scenario':<18} {'Active lane':<12} "
        f"{'Duration':<10} {'Action':<8} {'Status'}"
    )
    print("  " + "─" * 66)

    for desc, counts in test_cases:
        result = sc.decide(counts)
        errors = _validate(result, counts)

        dur_str = f"{result.get('duration', '?')} s"
        act_str = str(result.get("action", "?"))
        dir_str = result.get("direction", "?")

        if errors:
            status = f"❌  {errors[0]}"
            failed += 1
        else:
            status = "✓"
            passed += 1

        print(
            f"  {desc:<18} {dir_str:<12} {dur_str:<10} {act_str:<8} {status}"
        )

    # ── duration range verification ────────────────────────────────────────────
    print("\n  Checking duration range across 50 random inputs …")
    import numpy as np
    rng = np.random.default_rng(0)
    range_ok = True
    for _ in range(50):
        rand_counts = {
            k: int(rng.integers(0, 21)) for k in LANE_KEYS
        }
        result = sc.decide(rand_counts)
        dur = result.get("duration", -1)
        if not (MIN_GREEN <= dur <= MAX_GREEN):
            print(f"  ❌  duration {dur} out of range for {rand_counts}")
            range_ok = False
            failed += 1
    if range_ok:
        print(f"  ✓  All durations in [{MIN_GREEN}, {MAX_GREEN}] seconds")
        passed += 1

    # ── encode/decode roundtrip ────────────────────────────────────────────────
    print("\n  Checking encode/decode roundtrip for all 224 actions …")
    roundtrip_ok = True
    for a in range(ACTION_SIZE):
        d, dur = decode_action(a)
        if encode_action(d, dur) != a:
            print(f"  ❌  roundtrip failed for action {a}")
            roundtrip_ok = False
            failed += 1
            break
    if roundtrip_ok:
        print("  ✓  All 224 actions encode ↔ decode correctly")
        passed += 1

    # ── proportional fallback ──────────────────────────────────────────────────
    print("\n  Testing proportional fallback (no weights file) …")
    sc_fb = SignalController(
        weights_path=ROOT / "models" / "__no_weights__.pt")
    result = sc_fb.decide({"laneN": 8, "laneS": 2, "laneE": 5, "laneW": 1})
    errors = _validate(
        result, {"laneN": 8, "laneS": 2, "laneE": 5, "laneW": 1})
    if errors or sc_fb.mode != "proportional":
        print(f"  ❌  Proportional fallback failed: {errors or 'wrong mode'}")
        failed += 1
    else:
        print(
            f"  ✓  Proportional fallback: "
            f"dir={result['direction']}, dur={result['duration']}s"
        )
        passed += 1

    # ── summary ────────────────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print(f"  Result: {passed} passed, {failed} failed")
    print(f"{'═'*70}\n")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, default=None,
                   help="Path to trained .pt weights file")
    args = p.parse_args()
    run_tests(weights_path=args.weights)
