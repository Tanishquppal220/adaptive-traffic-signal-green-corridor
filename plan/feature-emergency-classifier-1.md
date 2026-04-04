---
goal: Ambulance-Focused Emergency Classifier Retraining and F1 Optimization Plan
version: 1.0
date_created: 2026-04-04
last_updated: 2026-04-04
owner: Adaptive Signal ML Team
status: 'In Progress'
tags: [feature, training, ml, classification, evaluation, runtime]
---

# Introduction

![Status: Planned](https://img.shields.io/badge/status-Planned-blue)

This plan defines deterministic steps to retrain the emergency vehicle classifier for ambulance-first runtime policy, eliminate split leakage risk, and maximize F1 score using threshold calibration instead of top-1 accuracy alone.

## 1. Requirements & Constraints

- **REQ-001**: THE SYSTEM SHALL trigger green-corridor policy only for ambulance-class predictions at runtime.
- **REQ-002**: THE SYSTEM SHALL preserve multi-class classifier outputs for diagnostics while enforcing ambulance-only policy decisioning.
- **REQ-003**: THE SYSTEM SHALL use a scene-independent and near-duplicate-resistant train/val/test split.
- **REQ-004**: THE SYSTEM SHALL optimize and report F1 score for ambulance as the positive class.
- **REQ-005**: THE SYSTEM SHALL select a confidence threshold from validation data by maximizing F1.
- **SEC-001**: THE SYSTEM SHALL not fetch untrusted remote artifacts at runtime; model and threshold artifacts SHALL be loaded from controlled local paths.
- **MLR-001**: Training SHALL remain reproducible with fixed seeds and deterministic split artifacts.
- **CON-001**: Existing runtime emergency integration contract in control/emergency_classifier.py and control/model_controller.py SHALL remain backward compatible.
- **CON-002**: Existing model filename models/emergency_vehicle_cls_yolov8s.pt SHALL remain valid unless an explicit migration task is completed.
- **GUD-001**: Hyperparameters and runtime thresholds SHALL be centralized in config.py.
- **PAT-001**: Validation metrics SHALL include class-wise precision, recall, F1, confusion matrix, and PR-curve threshold selection.

## 2. Implementation Steps

### Implementation Phase 1

- GOAL-001: Align runtime decision policy to ambulance-only triggering while preserving classifier diagnostics.

| Task     | Description | Completed | Date |
| -------- | ----------- | --------- | ---- |
| TASK-001 | In config.py, set EMERGENCY_LABEL_KEYWORDS to ambulance-only semantics and add explicit constant EMERGENCY_TARGET_LABEL = "ambulance". | x | 2026-04-04 |
| TASK-002 | In control/emergency_classifier.py, update_is_emergency() to perform exact-match against EMERGENCY_TARGET_LABEL before fallback keyword checks. | x | 2026-04-04 |
| TASK-003 | In control/emergency_classifier.py status(), expose active threshold and target label for runtime observability. | x | 2026-04-04 |
| TASK-004 | In control/test_runtime_contracts.py, add negative-case assertions where non-ambulance labels do not trigger emergency override. | x | 2026-04-04 |

### Implementation Phase 2

- GOAL-002: Build leakage-resistant and reproducible dataset splits for retraining.

| Task     | Description | Completed | Date |
| -------- | ----------- | --------- | ---- |
| TASK-005 | In training/emergency_vehicle_classifier_complete.ipynb dataset-prep cell, add hard cleanup of /content/data/dataset before creating split folders to prevent stale-file contamination. | x | 2026-04-04 |
| TASK-006 | In training/emergency_vehicle_classifier_complete.ipynb, add deterministic split ratios TRAIN_RATIO=0.70, VAL_RATIO=0.15, TEST_RATIO=0.15 and write to train/, val/, and test/ directories. | x | 2026-04-04 |
| TASK-007 | In training/emergency_vehicle_classifier_complete.ipynb, implement near-duplicate control using perceptual hash grouping (pHash) and enforce group-level split assignment. | x | 2026-04-04 |
| TASK-008 | In training/emergency_vehicle_classifier_complete.ipynb, persist split manifest CSV with columns image_path,class_label,split,hash_group under /content/data/dataset/split_manifest.csv. | x | 2026-04-04 |
| TASK-009 | In training/emergency_vehicle_classifier_complete.ipynb, add leakage assertion that no hash_group appears in more than one split. | x | 2026-04-04 |

### Implementation Phase 3

- GOAL-003: Replace overfit-prone defaults with F1-oriented training configuration.

| Task     | Description | Completed | Date |
| -------- | ----------- | --------- | ---- |
| TASK-010 | In training/emergency_vehicle_classifier_complete.ipynb, remove fixed MAX_IMAGES=1000 cap from final training path and make cap optional via USE_IMAGE_CAP flag defaulting to False. | x | 2026-04-04 |
| TASK-011 | In training/emergency_vehicle_classifier_complete.ipynb, update train args to regularized defaults: freeze=6, dropout=0.2, weight_decay=0.001, label_smoothing=0.05, epochs=120, patience=20, lr0=0.0005. | x | 2026-04-04 |
| TASK-012 | In training/emergency_vehicle_classifier_complete.ipynb, add run-name parameterization (name=f"ambulance_f1_{timestamp}") so baseline and regularized experiments are comparable and not overwritten. | x | 2026-04-04 |
| TASK-013 | In training/emergency_vehicle_classifier_complete.ipynb, execute two fixed experiments (baseline and regularized) and store each run path and metrics in a results summary table. |  |  |

### Implementation Phase 4

- GOAL-004: Calibrate operating threshold by maximizing validation F1 for ambulance class.

| Task     | Description | Completed | Date |
| -------- | ----------- | --------- | ---- |
| TASK-014 | In training/emergency_vehicle_classifier_complete.ipynb, add inference-eval cell that iterates val split, captures ambulance probability, and stores y_true/y_score arrays. | x | 2026-04-04 |
| TASK-015 | In training/emergency_vehicle_classifier_complete.ipynb, compute precision-recall curve and threshold sweep to select threshold that maximizes F1 (argmax over thresholds). | x | 2026-04-04 |
| TASK-016 | In training/emergency_vehicle_classifier_complete.ipynb, compute and print confusion matrix, precision, recall, F1 at selected threshold and at default threshold=0.5 for comparison. | x | 2026-04-04 |
| TASK-017 | In training/emergency_vehicle_classifier_complete.ipynb, save calibration artifact /content/models/emergency_classifier_threshold.json containing selected threshold and validation metrics. | x | 2026-04-04 |

### Implementation Phase 5

- GOAL-005: Integrate calibrated threshold into runtime path without contract breaks.

| Task     | Description | Completed | Date |
| -------- | ----------- | --------- | ---- |
| TASK-018 | In config.py, add EMERGENCY_THRESHOLD_PATH defaulting to models/emergency_classifier_threshold.json and fallback behavior when file is missing. | x | 2026-04-04 |
| TASK-019 | In control/emergency_classifier.py, load calibrated threshold from EMERGENCY_THRESHOLD_PATH at startup and fallback to EMERGENCY_CONFIDENCE_THRESHOLD if unavailable. | x | 2026-04-04 |
| TASK-020 | In control/emergency_classifier.py classify(), keep predictions payload unchanged but gate detected=True strictly by ambulance target label and calibrated threshold. | x | 2026-04-04 |

### Implementation Phase 6

- GOAL-006: Validate retraining quality and runtime safety using deterministic checks.

| Task     | Description | Completed | Date |
| -------- | ----------- | --------- | ---- |
| TASK-021 | Add test script control/test_emergency_classifier_policy.py to verify ambulance-only trigger behavior and threshold gating using mocked classifier outputs. | x | 2026-04-04 |
| TASK-022 | Execute validation commands: uv run ruff check control config.py and uv run python control/test_runtime_contracts.py and uv run python control/test_emergency_classifier_policy.py. | x | 2026-04-04 |
| TASK-023 | Run notebook validation on test split and record final metrics table containing val_f1, test_f1, val_precision, val_recall, threshold, and selected checkpoint path. |  |  |

## 3. Alternatives

- **ALT-001**: Keep current random 80/20 split without duplicate controls. Rejected because leakage can inflate validation metrics and hide overfitting.
- **ALT-002**: Optimize top-1 accuracy only and use fixed threshold 0.5. Rejected because runtime requirement is policy-quality ambulance triggering and user requested F1 optimization.
- **ALT-003**: Trigger runtime preemption for any emergency type (ambulance/fire/police). Rejected because runtime UI and policy target is ambulance-only response.

## 4. Dependencies

- **DEP-001**: ultralytics for classifier training and inference.
- **DEP-002**: scikit-learn for precision-recall curve, F1, and confusion matrix computations.
- **DEP-003**: imagehash and Pillow for perceptual-hash duplicate grouping.
- **DEP-004**: Existing runtime integration in control/emergency_classifier.py and control/model_controller.py.

## 5. Files

- **FILE-001**: training/emergency_vehicle_classifier_complete.ipynb - dataset split hardening, regularized training, F1 calibration, and test metrics reporting.
- **FILE-002**: config.py - ambulance target label and threshold artifact path constants.
- **FILE-003**: control/emergency_classifier.py - ambulance-only gating and calibrated threshold loading.
- **FILE-004**: control/test_runtime_contracts.py - emergency policy regression checks.
- **FILE-005**: control/test_emergency_classifier_policy.py - new dedicated policy behavior tests.
- **FILE-006**: README.md - update model training and runtime emergency policy documentation.

## 6. Testing

- **TEST-001**: Data split integrity test - assert no hash-group leakage across train/val/test.
- **TEST-002**: Metric reproducibility test - rerun notebook split with same seed and verify identical split_manifest.csv.
- **TEST-003**: Threshold optimization test - verify selected threshold equals argmax-F1 from validation sweep.
- **TEST-004**: Runtime policy test - verify non-ambulance predictions above threshold do not set detected=True.
- **TEST-005**: End-to-end smoke test - run uv run python control/test_controller.py and verify no contract regression.

## 7. Risks & Assumptions

- **RISK-001**: Ambulance-only target may reduce recall for edge-case emergency vehicles mislabeled as other classes.
- **RISK-002**: Aggressive regularization may underfit if class diversity remains low after leakage controls.
- **RISK-003**: pHash grouping may over-cluster visually similar but distinct scenes if distance threshold is too high.
- **ASSUMPTION-001**: Existing dataset contains a valid ambulance class label and sufficient positive samples for F1 optimization.
- **ASSUMPTION-002**: Runtime preemption decisions should prioritize balanced precision/recall as requested, not asymmetric cost weighting.
- **ASSUMPTION-003**: Notebook retraining workflow remains the primary model build path for this repository.

## 8. Related Specifications / Further Reading

- [plan/refactor-runtime-bugfixes-1.md](refactor-runtime-bugfixes-1.md)
- [AGENTS.md](../AGENTS.md)
- [README.md](../README.md)
- [control/emergency_classifier.py](../control/emergency_classifier.py)
- [training/emergency_vehicle_classifier_complete.ipynb](../training/emergency_vehicle_classifier_complete.ipynb)
