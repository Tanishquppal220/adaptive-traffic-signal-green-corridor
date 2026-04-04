---
goal: Runtime Bugfix and Hardening Plan for Decision, API, and Synthetic Runtime
version: 1.0
date_created: 2026-04-04
last_updated: 2026-04-04
owner: Traffic Control Platform Team
status: 'Completed'
tags: [bug, refactor, reliability, security, api, runtime]
---

# Introduction

![Status: Completed](https://img.shields.io/badge/status-Completed-brightgreen)

This plan defines deterministic implementation steps to fix identified correctness, security, and reliability bugs in decision orchestration, API input handling, synthetic runtime state management, and package/runtime configuration.

## 1. Requirements & Constraints

- **REQ-001**: THE SYSTEM SHALL ensure `action` always encodes the final executed `(direction, duration)` pair after emergency, fairness, scheduler, and low-traffic overrides.
- **REQ-002**: THE SYSTEM SHALL return HTTP `400` with structured error payloads for invalid synthetic API numeric inputs instead of raising unhandled exceptions.
- **REQ-003**: THE SYSTEM SHALL protect synthetic runtime state updates with lock-based synchronization to prevent cross-request race conditions.
- **REQ-004**: THE SYSTEM SHALL use a single canonical lane-count normalization path from `control/schema.py` across controller and route layers.
- **REQ-005**: THE SYSTEM SHALL keep DQN fallback path resolution deterministic in training scripts across invocation contexts.
- **REQ-006**: THE SYSTEM SHALL keep dependency declarations deduplicated and valid in project metadata.
- **SEC-001**: THE SYSTEM SHALL run Flask debug mode disabled by default and bind to localhost by default unless explicitly overridden.
- **CON-001**: Existing response key schema for `/api/run_cycle`, `/api/synthetic_cycle`, and `/api/spawn_ambulance` SHALL remain backward compatible.
- **CON-002**: Existing model unavailability fallback behavior (detector/classifier/predictor/controller) SHALL be preserved.
- **CON-003**: Existing action codec contract from `training/DQN/environment.py` SHALL remain the single source of truth.
- **GUD-001**: Configuration values SHALL be centralized in `config.py`; hardcoded route-level magic numbers SHALL be avoided.
- **PAT-001**: Decision mutation paths SHALL recompute action using `encode_action()` immediately after direction or duration mutation.
- **PAT-002**: Shared mutable runtime state SHALL be accessed through explicit lock-guarded helpers.

## 2. Implementation Steps

### Implementation Phase 1

- GOAL-001: Enforce action-direction-duration consistency across all executed decision paths.

| Task     | Description | Completed | Date |
| -------- | ----------- | --------- | ---- |
| TASK-001 | In `control/model_controller.py`, add helper `_encode_decision_action(direction: str, duration: int) -> int` using `DIRECTIONS`, duration clamp to `[cfg.MIN_GREEN, cfg.MAX_GREEN]`, and `encode_action`. | x | 2026-04-04 |
| TASK-002 | In `ModelController.decide_from_lane_counts()`, update emergency override branch to set `action` from `_encode_decision_action(emergency_direction, emergency_duration)`. | x | 2026-04-04 |
| TASK-003 | In `ModelController._apply_fairness_policy()`, update hard and soft override decisions to set `action` from `_encode_decision_action(direction, baseline_duration)`. | x | 2026-04-04 |
| TASK-004 | In `gui/routes.py`, update `_apply_low_traffic_duration_policy()`, `_apply_fixed_round_robin_override()`, and `_apply_non_repeat_scheduler()` to recompute `action` after any direction or duration change. | x | 2026-04-04 |

Completion Criteria: `decode_action(result["action"])` matches API-reported direction and duration for all override modes.

### Implementation Phase 2

- GOAL-002: Harden request parsing and eliminate normalization drift.

| Task     | Description | Completed | Date |
| -------- | ----------- | --------- | ---- |
| TASK-005 | In `gui/routes.py`, remove local `_normalize_lane_counts()` implementation and import/use `normalize_lane_counts` from `control/schema.py`. | x | 2026-04-04 |
| TASK-006 | In `gui/routes.py`, add `_parse_float_field()` and `_parse_int_field()` helpers that validate type conversion and enforce bounds for `intensity`, `seed`, and `tick`. | x | 2026-04-04 |
| TASK-007 | In `synthetic_cycle()` and `spawn_ambulance()`, wrap body parsing in `try/except ValueError` and return `jsonify({"error": ...}), 400` on invalid payloads. | x | 2026-04-04 |
| TASK-008 | In `config.py`, add synthetic validation bounds constants (`SYNTHETIC_INTENSITY_MIN`, `SYNTHETIC_INTENSITY_MAX`, `SYNTHETIC_SEED_MIN`, `SYNTHETIC_TICK_MIN`) and consume them in `gui/routes.py`. | x | 2026-04-04 |

Completion Criteria: Invalid synthetic payloads never produce HTTP `500` and always produce deterministic HTTP `400` responses.

### Implementation Phase 3

- GOAL-003: Make synthetic runtime state updates concurrency-safe.

| Task     | Description | Completed | Date |
| -------- | ----------- | --------- | ---- |
| TASK-009 | In `gui/routes.py`, introduce `SYNTHETIC_RUNTIME_LOCK = threading.Lock()` and lock-guarded helpers `_read_synthetic_runtime_snapshot()` and `_update_synthetic_runtime(fields: dict[str, Any])`. | x | 2026-04-04 |
| TASK-010 | Refactor `synthetic_cycle()` to read runtime snapshot and apply writes via lock-guarded helpers, replacing direct dictionary mutations. | x | 2026-04-04 |
| TASK-011 | Refactor `spawn_ambulance()` and `synthetic_reset()` to perform all `SYNTHETIC_RUNTIME` writes under lock. | x | 2026-04-04 |
| TASK-012 | Add deterministic comments in `gui/routes.py` documenting lock invariants for runtime mutation boundaries. | x | 2026-04-04 |

Completion Criteria: No direct unlocked writes remain for `SYNTHETIC_RUNTIME` in route handlers.

### Implementation Phase 4

- GOAL-004: Apply secure runtime defaults for Flask execution.

| Task     | Description | Completed | Date |
| -------- | ----------- | --------- | ---- |
| TASK-013 | In `config.py`, replace hardcoded `FLASK_DEBUG` and `FLASK_HOST` with environment-derived values using secure defaults (`False`, `127.0.0.1`). | x | 2026-04-04 |
| TASK-014 | In `main.py`, keep `app.run()` invocation but set `use_reloader=FLASK_DEBUG` and preserve host/port from config. | x | 2026-04-04 |
| TASK-015 | In `README.md`, update run instructions to show explicit local debug enable command and safe default behavior note. | x | 2026-04-04 |

Completion Criteria: Default launch does not expose debug mode on `0.0.0.0`.

### Implementation Phase 5

- GOAL-005: Resolve packaging and training-path hygiene issues.

| Task     | Description | Completed | Date |
| -------- | ----------- | --------- | ---- |
| TASK-016 | In `training/DQN/train.py`, set fallback `ROOT` to repository root (`Path(__file__).resolve().parents[2]`) and keep fallback model path under `ROOT / "models"`. | x | 2026-04-04 |
| TASK-017 | In `pyproject.toml`, remove duplicate `scikit-learn` and `xgboost` entries and remove invalid `tk>=0.1.0` dependency. | x | 2026-04-04 |
| TASK-018 | Regenerate dependency lock metadata with `uv lock` after dependency cleanup and verify install with `uv sync`. | x | 2026-04-04 |

Completion Criteria: Dependency list has unique entries and training fallback model path resolves correctly from DQN module execution.

### Implementation Phase 6

- GOAL-006: Add regression checks and complete validation.

| Task     | Description | Completed | Date |
| -------- | ----------- | --------- | ---- |
| TASK-019 | Add new script `control/test_runtime_contracts.py` to assert `decode_action(action)` equals reported direction/duration for emergency, fairness, scheduler, and low-traffic decisions. | x | 2026-04-04 |
| TASK-020 | Update `control/test_controller.py` to include explicit consistency assertion for action-direction-duration coupling in all existing scenarios. | x | 2026-04-04 |
| TASK-021 | Execute validation commands: `uv run ruff check control gui training`, `uv run python control/test_controller.py`, and `uv run python control/test_runtime_contracts.py`. | x | 2026-04-04 |
| TASK-022 | Execute API error-handling smoke checks (invalid `intensity`, `seed`, and `tick`) and verify HTTP `400` contract with stable error messages. | x | 2026-04-04 |

Completion Criteria: All planned tests pass and synthetic API error cases return deterministic HTTP `400` responses.

## 3. Alternatives

- **ALT-001**: Keep original DQN `action` unchanged and add separate `executed_action` field. Rejected because current `online_update()` flow expects a single action semantic for executed control.
- **ALT-002**: Introduce Pydantic request schemas for all Flask endpoints. Rejected for this iteration to avoid dependency and framework expansion in a targeted bugfix release.
- **ALT-003**: Move synthetic runtime state to Redis for concurrency safety. Rejected for this iteration because deployment complexity is unnecessary for local/demo runtime scope.

## 4. Dependencies

- **DEP-001**: Existing codec functions `encode_action` and `decode_action` in `training/DQN/environment.py`.
- **DEP-002**: Existing Flask route stack in `gui/routes.py` and shared schema utilities in `control/schema.py`.
- **DEP-003**: Python standard library synchronization primitives (`threading.Lock`) for runtime-state protection.
- **DEP-004**: `uv` toolchain for dependency lock regeneration and environment synchronization.

## 5. Files

- **FILE-001**: `control/model_controller.py` - decision override action recomputation and helper introduction.
- **FILE-002**: `gui/routes.py` - scheduler/low-traffic action recomputation, parsing validation, runtime lock protection, normalization consolidation.
- **FILE-003**: `control/schema.py` - canonical normalization usage source (consumed by routes).
- **FILE-004**: `config.py` - secure Flask defaults and synthetic input bounds constants.
- **FILE-005**: `main.py` - secure `app.run` behavior refinement.
- **FILE-006**: `training/DQN/train.py` - robust fallback root/model path resolution.
- **FILE-007**: `pyproject.toml` - dependency deduplication and invalid dependency removal.
- **FILE-008**: `control/test_controller.py` - strengthened action contract assertions.
- **FILE-009**: `control/test_runtime_contracts.py` - new regression script for override contract validation.
- **FILE-010**: `README.md` - secure run-mode documentation updates.

## 6. Testing

- **TEST-001**: Static checks - run `uv run ruff check control gui training` and require zero new lint violations in modified files.
- **TEST-002**: Controller integration - run `uv run python control/test_controller.py` and require full pass.
- **TEST-003**: Override contract regression - run `uv run python control/test_runtime_contracts.py` and require full pass.
- **TEST-004**: API validation regression - send malformed JSON fields to `/api/synthetic_cycle` and `/api/spawn_ambulance`; require HTTP `400` with non-empty `error` message.
- **TEST-005**: Manual runtime check - run `uv run python main.py` and verify default startup is localhost with debug disabled unless explicitly configured.

## 7. Risks & Assumptions

- **RISK-001**: Tightening input parsing may reject previously tolerated malformed payloads from custom clients.
- **RISK-002**: Runtime locking may reduce synthetic endpoint throughput under high parallel request load.
- **RISK-003**: Dependency cleanup may require lockfile refresh and environment resync in existing developer setups.
- **ASSUMPTION-001**: API consumers depend on existing response schema fields and not on malformed payload permissiveness.
- **ASSUMPTION-002**: The DQN action codec contract (`direction`, `duration` mapping) remains unchanged during this bugfix cycle.
- **ASSUMPTION-003**: Current deployment scope is demo/single-instance Flask where lock-based in-process synchronization is sufficient.

## 8. Related Specifications / Further Reading

- [AGENTS.md](../AGENTS.md)
- [.github/copilot-instructions.md](../.github/copilot-instructions.md)
- [README.md](../README.md)
- [docs/PROJECT_NOTES.md](../docs/PROJECT_NOTES.md)
- [docs/project-blueprin.md](../docs/project-blueprin.md)
