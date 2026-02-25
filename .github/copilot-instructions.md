# Project Guidelines

## Code Style
- Mirror the lightweight structure of [main.py](../main.py): keep each module focused, use 4-space indentation, and prefer concise, descriptive helper functions so the entrypoint stays easy to follow.
- Use explicit type hints on public interfaces (e.g., the `main()` stub) and add docstrings when a function encapsulates more than a few lines of logic.
- Keep imports grouped (standard library, third-party, local) and sorted alphabetically; add new dependencies through [pyproject.toml](../pyproject.toml).

## Architecture
- The control loop relies on five model families: YOLOv8 for general vehicle detection, LSTM/XGBoost for traffic-density prediction, Q-learning/DQN for signal timing, a CNN audio classifier for siren recognition, and a fine-tuned YOLOv8 variant for ambulance detection.
- Design each model as its own orchestrated component that feeds a central signal optimizer; detection outputs should stream into the density predictor, which in turn drives the adaptive timing logic.
- Keep the control-layer glue code separate from model implementations so future telemetry (e.g., `main.py` or a dedicated controller module) can swap sensors and optimizers without affecting inference code.

## Build and Test
- `pip install -e .` to register the project with the local environment before modifying modules.
- `python main.py` currently exercises the bare entrypoint and verifies the packaging setup; extend it into a richer CLI or service once the models become available.
- No automated tests are defined yet; if you add tests, register them in the `tool.<name>` section of [pyproject.toml](../pyproject.toml) and run them via `python -m <test_runner>`.

## Project Conventions
- Place each subsystem (detection, prediction, optimization, audio, ambulance) in its own package/directory; keep configuration values centralized so that new models just import the shared config instead of duplicating constants.
- Store long-lived assets (weights, scene layouts) in a `models/` or `assets/` directory and reference them relative to the workspace root to keep paths resolvable from both CLI and service deployments.
- Update [pyproject.toml](../pyproject.toml) with any new runtime dependencies or build metadata instead of copying requirements files.

## Integration Points
- Camera streams feed YOLOv8 detectors; their bounding-box outputs should be passed immediately to the density predictor and the Q-learning agent.
- The CNN siren classifier monitors audio sources and should emit priority flags that the timing optimizer can consume alongside vehicle density.
- The ambulance-dedicated YOLOv8 variant shares the same camera inputs but writes to a separate channel so its detections can preempt normal timing cycles.

## Security
- No secrets live in the repo today; if you introduce keys (camera credentials, cloud services), load them from environment variables and document the required names in this file.
- Validate any external data coming from sensors/cameras before trusting it in the optimizer so attackers cannot spoof signal decisions.

Please let me know if any section is unclear or missing so I can iterate further on these instructions.