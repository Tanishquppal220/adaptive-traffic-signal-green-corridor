"""
Simulation engine for the adaptive traffic signal system.

Central state machine that manages all simulation state and advances it on a fixed tick.
Thread-safe design allows Flask routes to read snapshots while the engine ticks in background.

**ML Integration**:
- Uses TrafficDensityPredictor (XGBoost) to predict future lane densities
- Predictions are used to optimize signal timing (prioritize lanes with predicted congestion)
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass
from typing import Literal

DIRECTIONS = ["N", "S", "E", "W"]
DIRECTION_FULL = {"N": "North", "S": "South", "E": "East", "W": "West"}
OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}
SIGNAL_COLORS = Literal["GREEN", "YELLOW", "RED"]


@dataclass
class SimConfig:
    """All tunable simulation parameters in one place."""

    # Signal timing
    min_green_seconds: int = 10
    max_green_seconds: int = 60
    yellow_seconds: int = 4

    # Traffic generation (vehicles spawned per tick when auto-gen enabled)
    base_spawn_rate: float = 0.3
    rush_hour_multiplier: float = 2.5
    night_multiplier: float = 0.2

    # Vehicle departure rate (vehicles per green tick)
    departure_rate: int = 2

    # Emergency
    ambulance_speed: float = 0.05  # position delta per tick (0=far, 1=at intersection)
    corridor_timeout_seconds: int = 120

    # Simulation
    tick_interval_seconds: float = 1.0
    sim_speed: float = 1.0  # 1.0 = realtime, 2.0 = 2x speed

    # Auto vehicle generation
    auto_generate: bool = True

    # ML-based optimization
    use_ml_prediction: bool = True  # Use XGBoost predictor for signal timing


@dataclass
class LaneState:
    """State of a single lane (N, S, E, or W)."""

    direction: str
    vehicle_count: int = 0
    has_ambulance: bool = False
    wait_time_seconds: float = 0.0
    vehicles_passed_total: int = 0


@dataclass
class SignalState:
    """State of a traffic signal for one direction."""

    direction: str
    color: SIGNAL_COLORS = "RED"
    time_remaining: int = 30
    phase_duration: int = 30


@dataclass
class AmbulanceState:
    """State of the emergency vehicle (ambulance)."""

    active: bool = False
    entry_direction: str | None = None
    exit_direction: str | None = None
    position: float = 0.0  # 0.0 = far away, 1.0 = at intersection, 2.0 = exited
    speed: float = 0.05
    corridor_active: bool = False
    corridor_direction: str | None = None
    confidence: float = 0.0


# Preset traffic scenarios
SCENARIOS = {
    "normal": {
        "label": "Normal Traffic",
        "description": "Regular daytime traffic across all lanes",
        "initial_counts": {"N": 8, "S": 6, "E": 10, "W": 5},
        "spawn_rate": 0.25,  # 25% chance per tick per lane
    },
    "rush_hour": {
        "label": "Morning Rush Hour",
        "description": "High volume, heavy congestion on all lanes",
        "initial_counts": {"N": 20, "S": 18, "E": 25, "W": 15},
        "spawn_rate": 0.5,  # 50% chance per tick per lane (more vehicles appear)
    },
    "night": {
        "label": "Late Night",
        "description": "Sparse traffic, low density",
        "initial_counts": {"N": 2, "S": 1, "E": 3, "W": 1},
        "spawn_rate": 0.08,  # 8% chance - very few vehicles
    },
    "emergency_demo": {
        "label": "Emergency Demo",
        "description": "Moderate traffic with ambulance demonstration",
        "initial_counts": {"N": 12, "S": 10, "E": 12, "W": 8},
        "spawn_rate": 0.25,
        "auto_ambulance": {"entry": "N", "exit": "S"},
    },
    "imbalanced": {
        "label": "Imbalanced Load",
        "description": "One lane severely congested",
        "initial_counts": {"N": 35, "S": 3, "E": 5, "W": 4},
        "spawn_rate": 0.35,
    },
}


class SimulationEngine:
    """
    Central simulation engine. Owns world state and advances it on a fixed tick.

    Instantiate once, call engine.start() to begin background ticking.
    Read engine.get_snapshot() from Flask routes — it is always thread-safe.

    **ML Integration**:
    - TrafficDensityPredictor provides future density predictions
    - Predictions influence signal timing decisions
    """

    def __init__(self, config: SimConfig | None = None):
        self.config = config or SimConfig()
        self.tick_count = 0
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

        # World state
        self.lanes: dict[str, LaneState] = {d: LaneState(direction=d) for d in DIRECTIONS}
        self.signals: dict[str, SignalState] = {d: SignalState(direction=d) for d in DIRECTIONS}
        self.ambulance = AmbulanceState()
        self.active_scenario = "normal"
        self.mode: str = "NORMAL"
        self.event_log: list[str] = []

        # Signal FSM state
        self._phase_index = 0
        self._current_green_dir: str | None = "N"
        self._in_yellow = False
        self._yellow_countdown = 0

        # Metrics accumulator
        self._total_vehicles_passed = 0
        self._total_wait_time = 0.0
        self._start_time = time.time()

        # ML Predictor for traffic density forecasting
        self._predictor = None
        self._last_prediction: dict[str, float] = {}
        self._prediction_confidence: dict[str, float] = {}
        self._ml_available = False

        if self.config.use_ml_prediction:
            self._init_ml_predictor()

        # Initialize signals: North starts green
        self.signals["N"].color = "GREEN"
        self.signals["N"].time_remaining = 30
        self.signals["N"].phase_duration = 30
        for d in ["S", "E", "W"]:
            self.signals[d].color = "RED"
            self.signals[d].time_remaining = 30

    def _init_ml_predictor(self):
        """Initialize the ML traffic density predictor."""
        try:
            from detection.traffic_predictor import TrafficDensityPredictor

            self._predictor = TrafficDensityPredictor()
            self._ml_available = True
            self.log_event("🤖 ML Predictor loaded (XGBoost density forecasting)")
        except FileNotFoundError as e:
            self.log_event(f"⚠️ ML models not found: {e}")
            self._ml_available = False
        except Exception as e:
            self.log_event(f"⚠️ ML init failed: {e}")
            self._ml_available = False

    def start(self):
        """Start background tick thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._tick_loop, daemon=True)
        self._thread.start()
        self.log_event("🚦 Simulation started")

    def stop(self):
        """Stop the simulation tick loop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self.log_event("⏹️ Simulation stopped")

    def _tick_loop(self):
        """Background loop that calls _do_tick at the configured interval."""
        while self._running:
            tick_start = time.time()
            self._do_tick()
            elapsed = time.time() - tick_start
            sleep_time = (self.config.tick_interval_seconds / self.config.sim_speed) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _do_tick(self):
        """
        Execute one simulation tick. Order of operations:
        1. Traffic generation (if auto-gen enabled)
        2. ML prediction update (feed current counts to predictor)
        3. Signal FSM advancement (uses ML predictions for timing)
        4. Vehicle departure (drain green queues)
        5. Ambulance advancement
        6. Mode determination
        7. Wait time accumulation
        """
        with self._lock:
            self.tick_count += 1

            # 1. Auto-generate traffic if enabled
            if self.config.auto_generate:
                self._generate_traffic()

            # 2. Update ML predictor with current traffic state
            self._update_ml_prediction()

            # 3. Advance signal FSM
            self._advance_signals()

            # 4. Drain green queues
            self._drain_green_queues()

            # 5. Advance ambulance
            self._advance_ambulance()

            # 6. Update mode
            self._update_mode()

            # 7. Accumulate wait time for vehicles on red
            self._accumulate_wait_time()

            # 8. Trim event log
            if len(self.event_log) > 50:
                self.event_log = self.event_log[-50:]

    def _update_ml_prediction(self):
        """Feed current lane counts to ML predictor and get predictions."""
        if not self._ml_available or self._predictor is None:
            return

        # Update predictor history with current counts
        current = {d: float(self.lanes[d].vehicle_count) for d in DIRECTIONS}
        self._predictor.update_history(current)

        # Get prediction every 5 ticks (to avoid excessive computation)
        if self.tick_count % 5 == 0:
            try:
                prediction = self._predictor.predict(current)
                self._last_prediction = prediction.predicted_densities
                self._prediction_confidence = prediction.confidence_scores
            except Exception:
                pass  # Silently continue if prediction fails

    def _generate_traffic(self):
        """Probabilistically spawn vehicles in each lane."""
        for d in DIRECTIONS:
            if random.random() < self.config.base_spawn_rate:
                vehicles_added = random.choices([1, 2], weights=[0.8, 0.2])[0]
                self.lanes[d].vehicle_count = min(50, self.lanes[d].vehicle_count + vehicles_added)

    def _advance_signals(self):
        """Advance signal FSM: manage GREEN → YELLOW → RED transitions."""
        # Emergency corridor override takes priority
        if self.ambulance.corridor_active:
            self._apply_corridor_override()
            return

        current = self._current_green_dir
        if current is None:
            self._start_next_phase()
            return

        sig = self.signals[current]

        if self._in_yellow:
            self._yellow_countdown -= 1
            sig.time_remaining = self._yellow_countdown
            if self._yellow_countdown <= 0:
                sig.color = "RED"
                sig.time_remaining = 0
                self._in_yellow = False
                self._current_green_dir = None
                self._start_next_phase()
        else:
            sig.time_remaining -= 1
            if sig.time_remaining <= 0:
                # Transition to YELLOW
                sig.color = "YELLOW"
                self._in_yellow = True
                self._yellow_countdown = self.config.yellow_seconds
                sig.time_remaining = self.config.yellow_seconds

    def _start_next_phase(self):
        """Determine next direction and set it green using ML predictions."""
        # Use ML to select optimal next direction and duration
        if self._ml_available and self._last_prediction:
            next_dir, duration = self._ml_decide_phase()
        else:
            next_dir, duration = self._weighted_round_robin()

        self._current_green_dir = next_dir
        self._phase_index = DIRECTIONS.index(next_dir)
        self.signals[next_dir].color = "GREEN"
        self.signals[next_dir].time_remaining = duration
        self.signals[next_dir].phase_duration = duration

        method = "ML" if self._ml_available else "RR"
        self.log_event(f"🟢 {DIRECTION_FULL[next_dir]} → GREEN for {duration}s ({method})")

    def _ml_decide_phase(self) -> tuple[str, int]:
        """Use ML predictions to decide next phase.

        Strategy: Prioritize lanes where predicted density is HIGH
        (they will become congested soon, so clear them now).
        """
        current = {d: self.lanes[d].vehicle_count for d in DIRECTIONS}
        predicted = self._last_prediction

        # Score each lane: current count + predicted increase
        scores = {}
        for d in DIRECTIONS:
            curr = current.get(d, 0)
            pred = predicted.get(d, curr)
            # Weight lanes that are predicted to get worse
            predicted_change = pred - curr
            # Score = current vehicles + bonus for predicted growth
            scores[d] = curr + max(0, predicted_change) * 1.5

        # Select lane with highest score (most urgent)
        next_dir = max(scores, key=lambda d: scores[d])

        # Duration based on combined current + predicted load
        combined = current[next_dir] + predicted.get(next_dir, 0)
        total = sum(current.values()) + sum(predicted.values()) or 1
        proportion = combined / total

        duration = self.config.min_green_seconds + int(
            proportion * (self.config.max_green_seconds - self.config.min_green_seconds)
        )
        duration = max(self.config.min_green_seconds, min(self.config.max_green_seconds, duration))

        return next_dir, duration

    def _weighted_round_robin(self) -> tuple[str, int]:
        """Fallback: weighted round robin when ML is unavailable."""
        self._phase_index = (self._phase_index + 1) % 4
        next_dir = DIRECTIONS[self._phase_index]

        counts = {d: self.lanes[d].vehicle_count for d in DIRECTIONS}
        total = sum(counts.values()) or 1
        proportion = counts[next_dir] / total

        duration = self.config.min_green_seconds + int(
            proportion * (self.config.max_green_seconds - self.config.min_green_seconds)
        )
        duration = max(self.config.min_green_seconds, min(self.config.max_green_seconds, duration))

        return next_dir, duration

    def _apply_corridor_override(self):
        """Force corridor direction to GREEN, all others RED."""
        corridor_dir = self.ambulance.corridor_direction
        for d in DIRECTIONS:
            if d == corridor_dir:
                self.signals[d].color = "GREEN"
                # Show estimated time until ambulance clears (based on position)
                remaining = max(1, int((2.0 - self.ambulance.position) / self.ambulance.speed))
                self.signals[d].time_remaining = min(remaining, 30)
            else:
                self.signals[d].color = "RED"
                self.signals[d].time_remaining = self.signals[corridor_dir].time_remaining

    def _drain_green_queues(self):
        """Remove vehicles from lanes with GREEN signal."""
        for d, signal in self.signals.items():
            if signal.color == "GREEN" and self.lanes[d].vehicle_count > 0:
                # Gradual departure: 1-2 vehicles per tick
                departed = min(self.config.departure_rate, self.lanes[d].vehicle_count)
                self.lanes[d].vehicle_count -= departed
                self.lanes[d].vehicles_passed_total += departed
                self._total_vehicles_passed += departed

    def _advance_ambulance(self):
        """Move ambulance along its path and handle corridor activation."""
        amb = self.ambulance
        if not amb.active:
            return

        # Advance position
        amb.position += amb.speed * self.config.sim_speed

        # Phase 1: Approaching (0.0 → 1.0)
        if amb.position < 1.0:
            # Activate corridor when ambulance is close enough
            if amb.position > 0.5 and not amb.corridor_active:
                # Clear the entry lane so ambulance can enter intersection
                amb.corridor_active = True
                amb.corridor_direction = amb.entry_direction
                amb.confidence = 0.85 + random.uniform(0, 0.1)
                dir_name = DIRECTION_FULL[amb.entry_direction]
                self.log_event(f"🚨 Ambulance detected! Clearing {dir_name} lane")

        # Phase 2: At intersection - determine exit if not set
        elif 1.0 <= amb.position < 1.1:
            if amb.exit_direction is None:
                # Default to opposite direction (straight through)
                amb.exit_direction = OPPOSITE.get(amb.entry_direction, "S")
                self.log_event(f"🚦 Ambulance exiting → {DIRECTION_FULL[amb.exit_direction]}")
            # Keep entry direction green until ambulance fully exits
            # (ambulance needs the entry lane green to cross)

        # Phase 3: Exiting (1.1 → 2.0)
        # Keep corridor active on entry direction until ambulance clears

        # Phase 4: Gone (position >= 2.0)
        if amb.position >= 2.0:
            self.log_event("✅ Ambulance cleared — resuming normal mode")
            self.ambulance = AmbulanceState()  # Reset
            self._current_green_dir = None  # Force phase restart

    def _update_mode(self):
        """Determine current mode based on ambulance state."""
        if self.ambulance.corridor_active:
            self.mode = "CORRIDOR_ACTIVE"
        elif self.ambulance.active:
            self.mode = "EMERGENCY"
        else:
            self.mode = "NORMAL"

    def _accumulate_wait_time(self):
        """Increment wait time for vehicles stuck on RED."""
        for d, lane in self.lanes.items():
            if self.signals[d].color == "RED" and lane.vehicle_count > 0:
                lane.wait_time_seconds += 1.0
                self._total_wait_time += lane.vehicle_count

    def log_event(self, message: str):
        """Add timestamped event to log."""
        ts = time.strftime("%H:%M:%S")
        self.event_log.append(f"[{ts}] {message}")

    def get_snapshot(self) -> dict:
        """Thread-safe snapshot for Flask routes. Returns JSON-serializable dict."""
        with self._lock:
            elapsed = max(1, time.time() - self._start_time)
            avg_wait = self._total_wait_time / max(1, self._total_vehicles_passed)

            return {
                "tick": self.tick_count,
                "timestamp": time.time(),
                "mode": self.mode,
                "active_scenario": self.active_scenario,
                "sim_speed": self.config.sim_speed,
                "auto_generate": self.config.auto_generate,
                "lanes": {
                    d: {
                        "vehicle_count": lane.vehicle_count,
                        "has_ambulance": lane.has_ambulance,
                        "wait_time_seconds": round(lane.wait_time_seconds, 1),
                        "vehicles_passed_total": lane.vehicles_passed_total,
                    }
                    for d, lane in self.lanes.items()
                },
                "signals": {
                    d: {
                        "color": sig.color,
                        "time_remaining": sig.time_remaining,
                        "phase_duration": sig.phase_duration,
                    }
                    for d, sig in self.signals.items()
                },
                "ambulance": {
                    "active": self.ambulance.active,
                    "entry_direction": self.ambulance.entry_direction,
                    "exit_direction": self.ambulance.exit_direction,
                    "position": round(self.ambulance.position, 3),
                    "corridor_active": self.ambulance.corridor_active,
                    "corridor_direction": self.ambulance.corridor_direction,
                    "confidence": round(self.ambulance.confidence, 2),
                },
                "metrics": {
                    "total_vehicles_passed": self._total_vehicles_passed,
                    "avg_wait_time": round(avg_wait, 1),
                    "throughput_per_minute": round(self._total_vehicles_passed / (elapsed / 60), 1),
                },
                "ml": {
                    "available": self._ml_available,
                    "predictions": self._last_prediction,
                    "confidence": self._prediction_confidence,
                    "history_ready": (
                        self._predictor.history_ready() if self._predictor else False
                    ),
                },
                "event_log": list(self.event_log[-15:]),
            }

    # ── Control methods called from Flask routes ──────────────────────────

    def set_lane_counts(self, counts: dict[str, int]):
        """Set vehicle counts for each lane manually."""
        with self._lock:
            for d, count in counts.items():
                if d in self.lanes:
                    self.lanes[d].vehicle_count = max(0, min(50, count))
            self.log_event("📝 Lane counts updated manually")

    def trigger_ambulance(self, entry_direction: str, exit_direction: str | None = None):
        """Spawn an ambulance approaching from the given direction."""
        with self._lock:
            self.ambulance = AmbulanceState(
                active=True,
                entry_direction=entry_direction,
                exit_direction=exit_direction,
                position=0.0,
                speed=self.config.ambulance_speed,
            )
            self.log_event(f"🚑 Ambulance approaching from {DIRECTION_FULL[entry_direction]}")

    def set_scenario(self, scenario_name: str):
        """Load a preset traffic scenario."""
        with self._lock:
            if scenario_name not in SCENARIOS:
                return
            scenario = SCENARIOS[scenario_name]
            self.active_scenario = scenario_name

            for d in DIRECTIONS:
                self.lanes[d].vehicle_count = scenario["initial_counts"].get(d, 0)

            self.config.base_spawn_rate = scenario["spawn_rate"]
            self.log_event(f"📋 Scenario: {scenario['label']}")

            # Auto-trigger ambulance if specified
            if "auto_ambulance" in scenario:
                amb = scenario["auto_ambulance"]
                self.ambulance = AmbulanceState(
                    active=True,
                    entry_direction=amb["entry"],
                    exit_direction=amb.get("exit"),
                    position=0.0,
                    speed=self.config.ambulance_speed,
                )
                self.log_event(f"🚑 Ambulance auto-triggered from {DIRECTION_FULL[amb['entry']]}")

    def set_sim_speed(self, speed: float):
        """Set simulation speed multiplier."""
        with self._lock:
            self.config.sim_speed = max(0.25, min(5.0, speed))
            self.log_event(f"⏩ Speed set to {self.config.sim_speed}x")

    def toggle_auto_generate(self, enabled: bool):
        """Enable or disable automatic vehicle generation."""
        with self._lock:
            self.config.auto_generate = enabled
            status = "enabled" if enabled else "disabled"
            self.log_event(f"🚗 Auto-generation {status}")

    def reset(self):
        """Reset simulation to initial state."""
        with self._lock:
            self.__init__(SimConfig())
            self.log_event("🔄 Simulation reset")
