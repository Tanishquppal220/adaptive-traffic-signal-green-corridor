# AGENTS.md

## Purpose

This guide is for coding agents working in this repository.
Follow these commands and conventions to keep changes consistent and low-risk.

## Project Snapshot

- Language: Python 3.13 (`.python-version`)
- Environment/package tool: `uv`
- Linting: `ruff`
- Ruff config: line length `100`, target `py313`, checks `E`, `F`, `I`
- Runtime entrypoint: `main.py`
- Web app factory: `gui/__init__.py:create_app`
- Main orchestration path: `control/model_controller.py`
- RL/training path: `training/DQN/*`

## Rules Files In This Repo

- Copilot rules present: `.github/copilot-instructions.md`
- Cursor rules not present:
  - `.cursorrules` not found
  - `.cursor/rules/` not found

## High-Importance Rules From Copilot Instructions

- Use `uv` for execution in this repository.
- Prefer `uv run ruff check .` for lint validation.
- Do not mass-format or rewrite unrelated files to fix pre-existing issues.
- Keep schema boundaries explicit:
  - Direction keys: `"N"`, `"S"`, `"E"`, `"W"`
  - Lane keys: `"laneN"`, `"laneS"`, `"laneE"`, `"laneW"`
- Preserve explicit fallback behavior when ML models/weights are missing.
- Prefer centralized constants from `config.py` over duplicated literals.
- Keep action/state contracts synchronized between training and inference code.

## Environment Setup Commands

- Initial sync: `uv sync`
- Include dev group (ruff): `uv sync --group dev`
- Check interpreter: `uv run python --version`

## Build Commands

- There is no dedicated CI build pipeline in this repository.
- Packaging metadata exists in `pyproject.toml` (Hatchling backend).
- Optional package build command: `uv build`

## Run Commands

- Start the web server: `uv run python main.py`

## Lint Commands

- Full lint: `uv run ruff check .`
- Lint specific module: `uv run ruff check control/model_controller.py`
- Lint changed areas only: `uv run ruff check control gui training`

## Test Commands

- Main integration test script:
  - `uv run python control/test_controller.py`
- Run same test with explicit weights:
  - `uv run python control/test_controller.py --weights models/dqn_signal_optimizer.pt`

## Single-Test / Narrow-Validation Commands

There is no formal pytest suite right now. Use these focused checks:

- Action codec roundtrip:
  - `uv run python -c "from training.DQN.environment import encode_action, decode_action; a=encode_action(2,37); print(a, decode_action(a))"`
- One controller decision sample:
  - `uv run python -c "from control.signal_controller import SignalController; sc=SignalController(); print(sc.decide({'laneN':8,'laneS':2,'laneE':5,'laneW':1}))"`
- Lane-count normalization sample:
  - `uv run python -c "from control.schema import normalize_lane_counts; print(normalize_lane_counts({'N':3,'laneS':2}))"`

## Training Commands (RL)

- Smoke run:
  - `uv run python training/DQN/train.py --steps 500 --log-interval 100`
- Full CPU training:
  - `uv run python training/DQN/train.py --steps 100000 --device cpu`
- GPU training (if available):
  - `uv run python training/DQN/train.py --steps 100000 --device cuda`

## Code Style: Imports

- Use import groups in this order:
  1) standard library
  2) third-party packages
  3) local project imports
- Keep one blank line between import groups.
- Maintain ruff-isort compliance (`I` rules).
- Prefer explicit imports; avoid wildcard imports.
- Preserve existing guarded import patterns when modules support dual execution paths.

## Code Style: Formatting

- Keep lines at or below `100` chars unless unavoidable.
- Follow existing formatting patterns in each edited file.
- Keep diffs focused; avoid unrelated whitespace-only changes.
- Do not add new formatting tools or style systems unless requested.

## Code Style: Typing

- Use modern Python typing syntax (`dict[str, int]`, `str | None`).
- Keep `from __future__ import annotations` in files that already use it.
- Add/keep type hints on public functions and methods.
- Use `Path` for filesystem paths.
- Use `Any` only for dynamic third-party objects or flexible payload boundaries.

## Code Style: Naming

- Classes: `PascalCase`
- Functions/variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Keep domain terms consistent:
  - `direction` for `N/S/E/W`
  - `lane` for `laneN/laneS/laneE/laneW`
- Prefer explicit mapping helpers instead of inline key translation.

## Code Style: Error Handling

- Fail soft for unavailable models and missing optional weights.
- Return structured status/mode information instead of crashing where possible.
- Keep fallback mode explicit (`"proportional"`, `"unavailable"`, etc.).
- Raise `ValueError` for malformed API input and invalid request payloads.
- Log recoverable runtime issues with clear warnings/info.

## Data And Contract Conventions

- Use helpers in `control/schema.py` for key conversion and normalization.
- Do not silently mix direction-key and lane-key dictionaries.
- Keep API response keys stable unless task explicitly changes contracts.
- Use RL action codec helpers from `training/DQN/environment.py`:
  - `encode_action(...)`
  - `decode_action(...)`

## Configuration Conventions

- Add new tunables to `config.py`.
- Import constants from `config.py` rather than duplicating magic values.
- Preserve compatibility aliases if renaming existing configuration keys.

## Change-Scope Rules For Agents

- Modify only files required by the task.
- Do not refactor unrelated code in the same change.
- Preserve existing runtime fallback behavior.
- Keep diagnostics payloads and mode flags intact.
- Avoid broad rewrites when targeted edits are sufficient.

## Pre-Completion Checklist

- Run `uv run ruff check` on changed files (or full repo when practical).
- Run `uv run python control/test_controller.py` for controller changes.
- Run at least one narrow validation command for schema/action contract edits.
- Confirm no accidental lane-vs-direction key mismatches were introduced.
- Confirm newly added imports and typing pass ruff checks.

## Note On Instruction Drift

`.github/copilot-instructions.md` still references `RL_model/*` paths.
In the current repository layout, equivalent active paths are:

- `control/test_controller.py`
- `training/DQN/environment.py`
- `training/DQN/dqn_agent.py`
- `training/DQN/train.py`

Keep the intent of those instructions, but use current paths when editing.
