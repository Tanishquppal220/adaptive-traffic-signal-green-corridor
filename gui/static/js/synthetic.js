const laneKeys = ["laneN", "laneS", "laneE", "laneW"];
const laneLabels = {
  laneN: "North",
  laneS: "South",
  laneE: "East",
  laneW: "West",
};

const form = document.getElementById("syntheticForm");
const stepBtn = document.getElementById("stepBtn");
const autoBtn = document.getElementById("autoBtn");
const spawnBtn = document.getElementById("spawnBtn");
const resetBtn = document.getElementById("resetBtn");

const timeOfDayNode = document.getElementById("timeOfDay");
const intensityNode = document.getElementById("intensity");
const intensityValueNode = document.getElementById("intensityValue");
const seedNode = document.getElementById("seed");
const ambulanceLaneNode = document.getElementById("ambulanceLane");
const fairnessModeNode = document.getElementById("fairnessMode");
const autoModeNode = document.getElementById("autoMode");

const formMessage = document.getElementById("formMessage");
const simStatus = document.getElementById("simStatus");
const emergencyBanner = document.getElementById("emergencyBanner");
const eventLogNode = document.getElementById("eventLog");

const metricTotalQueueNode = document.getElementById("metricTotalQueue");
const metricBaselineNode = document.getElementById("metricBaseline");
const metricDqnNode = document.getElementById("metricDqn");
const metricImprovementNode = document.getElementById("metricImprovement");

const cycleLockedValueNode = document.getElementById("cycleLockedValue");
const cycleRemainingValueNode = document.getElementById("cycleRemainingValue");
const controlSourceValueNode = document.getElementById("controlSourceValue");
const dqnReranValueNode = document.getElementById("dqnReranValue");

const fairnessModeValueNode = document.getElementById("fairnessModeValue");
const fairnessAppliedValueNode = document.getElementById("fairnessAppliedValue");
const fairnessReasonValueNode = document.getElementById("fairnessReasonValue");
const fairnessSelectionValueNode = document.getElementById("fairnessSelectionValue");

const phaseLaneNode = document.getElementById("phaseLane");
const phaseSignalNode = document.getElementById("phaseSignal");
const phaseRemainingNode = document.getElementById("phaseRemaining");

const canvas = document.getElementById("intersectionCanvas");
const ctx = canvas.getContext("2d");

let animationId = 0;
let runToken = 0;
let lastTs = 0;
let syntheticTick = 0;
let autoFlowTimer = 0;
let autoFlowRunning = false;
let autoTickBusy = false;

const world = {
  counts: { laneN: 0, laneS: 0, laneE: 0, laneW: 0 },
  movingCars: [],
  sequence: [],
  sequenceCursor: 0,
  activeLane: null,
  signal: "red",
  phaseType: "idle",
  phaseTimeMs: 0,
  clearAccumulatorMs: 0,
  clearRateMin: 1,
  clearRateMax: 2,
  yellowMs: 900,
  assignedDurationMs: 10000,
  emergencyLane: null,
  emergencyActive: false,
  emergencyCarSpawned: false,
  controlLane: null,
  controlLocked: false,
  finished: false,
  seededRand: Math.random,
};

const events = [];

function mulberry32(seed) {
  let t = seed >>> 0;
  return function seeded() {
    t += 0x6d2b79f5;
    let x = Math.imul(t ^ (t >>> 15), 1 | t);
    x ^= x + Math.imul(x ^ (x >>> 7), 61 | x);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

function setMessage(text, isError = false) {
  formMessage.textContent = text || "";
  formMessage.style.color = isError ? "#ff7f7f" : "#7ee7a1";
}

function addEvent(text) {
  const ts = new Date().toLocaleTimeString();
  events.unshift(`[${ts}] ${text}`);
  while (events.length > 8) {
    events.pop();
  }
  eventLogNode.textContent = events.join("\n");
}

function renderCounts() {
  laneKeys.forEach((lane) => {
    const node = document.getElementById(`count-${lane}`);
    if (node) {
      node.textContent = String(Math.max(0, world.counts[lane] || 0));
    }
  });
}

function updatePhaseText() {
  phaseLaneNode.textContent = world.activeLane ? laneLabels[world.activeLane] : "-";
  phaseSignalNode.textContent = world.signal.toUpperCase();
  if (world.phaseType === "fixed-green" || world.phaseType === "yellow") {
    phaseRemainingNode.textContent = `${Math.max(0, Math.ceil(world.phaseTimeMs / 1000))}s`;
  } else {
    phaseRemainingNode.textContent = "-";
  }
}

function renderMetrics(metrics = {}) {
  metricTotalQueueNode.textContent = metrics.total_queue ?? "-";
  metricBaselineNode.textContent = metrics.baseline_clearance_seconds
    ? `${metrics.baseline_clearance_seconds}s`
    : "-";
  metricDqnNode.textContent = metrics.dqn_clearance_seconds
    ? `${metrics.dqn_clearance_seconds}s`
    : "-";
  metricImprovementNode.textContent = Number.isFinite(metrics.improvement_pct)
    ? `${metrics.improvement_pct}%`
    : "-";
}

function renderFairness(fairness = {}) {
  fairnessModeValueNode.textContent = fairness.mode || "-";
  fairnessAppliedValueNode.textContent = fairness.applied ? "YES" : "NO";
  fairnessReasonValueNode.textContent = fairness.reason || "-";

  const selected = fairness.selected_lane || "-";
  const baseline = fairness.baseline_lane || "-";
  fairnessSelectionValueNode.textContent = `${selected} / ${baseline}`;

  const laneState = fairness.lane_state || {};
  laneKeys.forEach((lane) => {
    const node = document.getElementById(`wait-${lane}`);
    if (!node) return;
    const state = laneState[lane] || {};
    const wait = Number(state.wait_seconds || 0).toFixed(0);
    const missed = Number(state.missed_turns || 0);
    node.textContent = `wait ${wait}s | missed ${missed}`;
  });
}

function renderCycleMeta(sim = {}) {
  if (cycleLockedValueNode) {
    cycleLockedValueNode.textContent = sim.cycle_locked ? "YES" : "NO";
  }
  if (cycleRemainingValueNode) {
    const remaining = Number(sim.cycle_remaining_sec ?? 0);
    cycleRemainingValueNode.textContent = Number.isFinite(remaining) ? `${Math.max(0, Math.ceil(remaining))}s` : "-";
  }
  if (controlSourceValueNode) {
    controlSourceValueNode.textContent = sim.control_source || "-";
  }
  if (dqnReranValueNode) {
    dqnReranValueNode.textContent = sim.dqn_reran_this_tick ? "YES" : "NO";
  }
}

function applyEmergencyVisualState(sim) {
  const emergencyActive = sim.emergency_status === "active" || sim.emergency_detected;
  world.emergencyActive = emergencyActive;
  world.emergencyLane = emergencyActive ? sim.selected_lane : null;
  world.emergencyCarSpawned = false;

  if (sim.emergency_status === "active") {
    emergencyBanner.classList.remove("hidden");
    emergencyBanner.textContent = sim.emergency_message || "Emergency corridor active.";
    return;
  }

  if (sim.emergency_status === "cleared") {
    emergencyBanner.classList.remove("hidden");
    emergencyBanner.textContent = sim.emergency_message || "Emergency corridor cleared.";
    return;
  }

  emergencyBanner.classList.add("hidden");
  emergencyBanner.textContent = "";
}

function randomInt(min, max) {
  return Math.floor(world.seededRand() * (max - min + 1)) + min;
}

function nextLaneWithCars() {
  if (!world.sequence.length) {
    return null;
  }
  for (let i = 0; i < world.sequence.length; i += 1) {
    const idx = (world.sequenceCursor + i) % world.sequence.length;
    const lane = world.sequence[idx];
    if ((world.counts[lane] || 0) > 0) {
      world.sequenceCursor = (idx + 1) % world.sequence.length;
      return lane;
    }
  }
  return null;
}

function laneGeometry() {
  const w = canvas.width;
  const h = canvas.height;
  const cx = w / 2;
  const cy = h / 2;
  const roadW = Math.min(w, h) * 0.38;
  const half = roadW / 2;

  const xN = cx - 18;
  const xS = cx + 18;
  const yE = cy - 18;
  const yW = cy + 18;

  const stopN = cy - half - 8;
  const stopS = cy + half + 8;
  const stopE = cx + half + 8;
  const stopW = cx - half - 8;

  return { cx, cy, roadW, half, xN, xS, yE, yW, stopN, stopS, stopE, stopW, w, h };
}

function queueCarPose(lane, idx, g) {
  const spacing = 28;
  if (lane === "laneN") return { x: g.xN, y: g.stopN - 22 - idx * spacing, a: Math.PI / 2 };
  if (lane === "laneS") return { x: g.xS, y: g.stopS + 22 + idx * spacing, a: -Math.PI / 2 };
  if (lane === "laneE") return { x: g.stopE + 22 + idx * spacing, y: g.yE, a: Math.PI };
  return { x: g.stopW - 22 - idx * spacing, y: g.yW, a: 0 };
}

function movingPath(lane, g) {
  if (lane === "laneN") return { sx: g.xN, sy: g.stopN - 6, ex: g.xN, ey: g.h + 40, a: Math.PI / 2 };
  if (lane === "laneS") return { sx: g.xS, sy: g.stopS + 6, ex: g.xS, ey: -40, a: -Math.PI / 2 };
  if (lane === "laneE") return { sx: g.stopE + 6, sy: g.yE, ex: -40, ey: g.yE, a: Math.PI };
  return { sx: g.stopW - 6, sy: g.yW, ex: g.w + 40, ey: g.yW, a: 0 };
}

function spawnMovingCar(lane, emergencyCar = false) {
  const palette = ["#2a9d8f", "#2b8bd3", "#915bb8", "#c77d2d", "#3c5fcb"];
  world.movingCars.push({
    lane,
    t: 0,
    speed: 0.35 + world.seededRand() * 0.25,
    color: emergencyCar ? "#d93025" : palette[randomInt(0, palette.length - 1)],
    emergencyCar,
  });
}

function drawCar(x, y, angle, color, emergencyCar = false) {
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(angle);
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.roundRect(-12, -7, 24, 14, 4);
  ctx.fill();
  ctx.fillStyle = "rgba(255,255,255,0.62)";
  ctx.fillRect(-7, -5, 14, 3);
  if (emergencyCar) {
    ctx.fillStyle = "#87cefa";
    ctx.fillRect(-3, -9, 6, 2);
  }
  ctx.restore();
}

function drawSignals(g) {
  const signals = {
    laneN: { x: g.xN + 30, y: g.stopN + 16 },
    laneS: { x: g.xS - 30, y: g.stopS - 16 },
    laneE: { x: g.stopE - 16, y: g.yE - 30 },
    laneW: { x: g.stopW + 16, y: g.yW + 30 },
  };

  laneKeys.forEach((lane) => {
    const p = signals[lane];
    const isActive = world.activeLane === lane;
    ctx.fillStyle = "#111";
    ctx.fillRect(p.x - 8, p.y - 22, 16, 42);

    const activeRed = !isActive || world.signal === "red";
    const activeYellow = isActive && world.signal === "yellow";
    const activeGreen = isActive && world.signal === "green";
    const lamps = [
      { y: -14, c: "#d64545", on: activeRed },
      { y: -2, c: "#d9a404", on: activeYellow },
      { y: 10, c: "#2f9e44", on: activeGreen },
    ];

    lamps.forEach((lamp) => {
      ctx.beginPath();
      ctx.arc(p.x, p.y + lamp.y, 4.2, 0, Math.PI * 2);
      ctx.fillStyle = lamp.on ? lamp.c : "#3d3d3d";
      ctx.fill();
    });
  });
}

function drawScene() {
  const g = laneGeometry();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#2d343a";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.fillStyle = "#40484f";
  ctx.fillRect(g.cx - g.half, 0, g.roadW, g.h);
  ctx.fillRect(0, g.cy - g.half, g.w, g.roadW);

  ctx.strokeStyle = "rgba(240, 240, 240, 0.65)";
  ctx.lineWidth = 2;
  ctx.setLineDash([10, 10]);
  ctx.beginPath();
  ctx.moveTo(g.cx, 0);
  ctx.lineTo(g.cx, g.h);
  ctx.moveTo(0, g.cy);
  ctx.lineTo(g.w, g.cy);
  ctx.stroke();
  ctx.setLineDash([]);

  if (world.emergencyLane) {
    const alpha = 0.22 + Math.sin(Date.now() / 180) * 0.08;
    ctx.fillStyle = `rgba(217, 55, 55, ${alpha})`;
    if (world.emergencyLane === "laneN") ctx.fillRect(g.cx - 36, 0, 30, g.stopN);
    if (world.emergencyLane === "laneS") ctx.fillRect(g.cx + 6, g.stopS, 30, g.h - g.stopS);
    if (world.emergencyLane === "laneE") ctx.fillRect(g.stopE, g.cy - 36, g.w - g.stopE, 30);
    if (world.emergencyLane === "laneW") ctx.fillRect(0, g.cy + 6, g.stopW, 30);
  }

  laneKeys.forEach((lane) => {
    const count = world.counts[lane] || 0;
    const renderCount = Math.min(count, 18);
    for (let i = 0; i < renderCount; i += 1) {
      const pose = queueCarPose(lane, i, g);
      drawCar(pose.x, pose.y, pose.a, "#4a90d9");
    }
  });

  world.movingCars.forEach((car) => {
    const path = movingPath(car.lane, g);
    const x = path.sx + (path.ex - path.sx) * car.t;
    const y = path.sy + (path.ey - path.sy) * car.t;
    drawCar(x, y, path.a, car.color, car.emergencyCar);
  });

  drawSignals(g);
}

function enterGreen(lane) {
  world.activeLane = lane;
  world.signal = "green";
  world.phaseType = "fixed-green";
  world.phaseTimeMs = world.assignedDurationMs;
  world.clearAccumulatorMs = 0;
  simStatus.textContent = `Green phase for ${laneLabels[lane]} (${Math.round(world.assignedDurationMs / 1000)}s)`;
  updatePhaseText();
}

function enterYellow() {
  world.signal = "yellow";
  world.phaseType = "yellow";
  world.phaseTimeMs = world.yellowMs;
  simStatus.textContent = `Switching from ${laneLabels[world.activeLane]}...`;
  updatePhaseText();
}

function completeSimulation() {
  world.signal = "red";
  world.activeLane = null;
  world.phaseType = "done";
  world.finished = true;
  simStatus.textContent = "Simulation finished: all lanes cleared.";
  updatePhaseText();
}

function tickPhase(dtMs) {
  if (world.phaseType === "done" || world.phaseType === "idle") return;

  if (world.signal === "green" && world.activeLane) {
    world.clearAccumulatorMs += dtMs;
    while (world.clearAccumulatorMs >= 1000) {
      world.clearAccumulatorMs -= 1000;
      const lane = world.activeLane;
      if ((world.counts[lane] || 0) > 0) {
        const remove = Math.min(world.counts[lane], randomInt(world.clearRateMin, world.clearRateMax));
        world.counts[lane] -= remove;
        for (let i = 0; i < remove; i += 1) {
          const emergencyCar =
            world.emergencyActive &&
            world.emergencyLane === lane &&
            !world.emergencyCarSpawned;
          if (emergencyCar) {
            world.emergencyCarSpawned = true;
          }
          spawnMovingCar(lane, emergencyCar);
        }
      }
    }
  }

  if (world.phaseType === "fixed-green") {
    world.phaseTimeMs -= dtMs;
    if (world.phaseTimeMs <= 0) enterYellow();
  } else if (world.phaseType === "yellow") {
    world.phaseTimeMs -= dtMs;
    if (world.phaseTimeMs <= 0) {
      const nextLane = nextLaneWithCars();
      if (nextLane) {
        enterGreen(nextLane);
      } else {
        completeSimulation();
      }
    }
  }

  renderCounts();
  updatePhaseText();
}

function tickMovingCars(dtMs) {
  world.movingCars.forEach((car) => {
    car.t += (car.speed * dtMs) / 1000;
  });

  const hadEmergencyCar = world.movingCars.some((car) => car.emergencyCar);
  world.movingCars = world.movingCars.filter((car) => car.t < 1.05);
  const hasEmergencyCar = world.movingCars.some((car) => car.emergencyCar);

  if (world.emergencyActive && hadEmergencyCar && !hasEmergencyCar) {
    world.emergencyActive = false;
    world.emergencyLane = null;
    emergencyBanner.classList.add("hidden");
    emergencyBanner.textContent = "";
  }
}

function animationLoop(ts) {
  if (runToken === 0) return;
  if (!lastTs) lastTs = ts;
  const dtMs = Math.min(50, ts - lastTs);
  lastTs = ts;

  tickPhase(dtMs);
  tickMovingCars(dtMs);
  drawScene();

  if (!world.finished || world.movingCars.length > 0) {
    animationId = window.requestAnimationFrame(animationLoop);
  }
}

function startSimulation(payload, source = "synthetic") {
  const sim = payload.simulation;
  const decision = payload.decision;

  world.counts = { ...sim.initial_counts };
  world.sequence = [...sim.sequence];
  world.sequenceCursor = 0;
  world.movingCars = [];
  world.clearRateMin = sim.clear_rate_min || 1;
  world.clearRateMax = sim.clear_rate_max || 2;
  world.yellowMs = sim.yellow_ms || 900;
  world.assignedDurationMs = (sim.selected_duration || 1) * 1000;
  world.seededRand = mulberry32(sim.seed || Date.now());
  world.finished = false;
  world.controlLane = sim.selected_lane || null;
  world.controlLocked = Boolean(sim.cycle_locked);
  applyEmergencyVisualState(sim);

  addEvent(
    `${source}: ${decision.direction} selected for ${decision.duration}s` +
      ` | source=${sim.control_source || "unknown"}` +
      (sim.lane_repeat_blocked ? ` | scheduler=${sim.scheduler_reason || "no_consecutive_same_lane"}` : "")
  );
  renderCounts();
  enterGreen(sim.selected_lane);
  if (Number.isFinite(Number(sim.cycle_remaining_sec)) && world.phaseType === "fixed-green") {
    world.phaseTimeMs = Math.max(0, Number(sim.cycle_remaining_sec) * 1000);
  }
  const activeLabel = world.activeLane ? laneLabels[world.activeLane] : "-";
  simStatus.textContent =
    `Cycle locked on ${activeLabel}. Models refreshed every ${sim.model_refresh_sec || sim.tick_interval_sec || 3}s.` +
    (sim.lane_repeat_blocked ? ` Scheduler enforced lane rotation.` : "");
  drawScene();
  renderCycleMeta(sim);

  if (animationId) window.cancelAnimationFrame(animationId);
  lastTs = 0;
  animationId = window.requestAnimationFrame(animationLoop);
}

function applySyntheticTick(payload, source = "synthetic") {
  const sim = payload.simulation || {};
  const needsNewCycle =
    Boolean(sim.dqn_reran_this_tick) ||
    sim.control_source === "emergency_override" ||
    !world.controlLocked ||
    !world.activeLane;

  if (needsNewCycle) {
    startSimulation(payload, source);
    return;
  }

  world.counts = { ...sim.initial_counts };
  world.clearRateMin = sim.clear_rate_min || world.clearRateMin;
  world.clearRateMax = sim.clear_rate_max || world.clearRateMax;
  world.controlLocked = Boolean(sim.cycle_locked);
  applyEmergencyVisualState(sim);

  if (world.phaseType === "fixed-green" && Number.isFinite(Number(sim.cycle_remaining_sec))) {
    world.phaseTimeMs = Math.max(0, Number(sim.cycle_remaining_sec) * 1000);
  }

  const activeLabel = world.activeLane ? laneLabels[world.activeLane] : "-";
  simStatus.textContent =
    `Cycle locked on ${activeLabel}. Models refreshed every ${sim.model_refresh_sec || sim.tick_interval_sec || 3}s.`;

  renderCounts();
  updatePhaseText();
  drawScene();
  renderCycleMeta(sim);
}

function resetWorld() {
  runToken = 0;
  if (animationId) window.cancelAnimationFrame(animationId);
  animationId = 0;
  lastTs = 0;

  world.counts = { laneN: 0, laneS: 0, laneE: 0, laneW: 0 };
  world.movingCars = [];
  world.sequence = [];
  world.sequenceCursor = 0;
  world.activeLane = null;
  world.signal = "red";
  world.phaseType = "idle";
  world.phaseTimeMs = 0;
  world.clearAccumulatorMs = 0;
  world.assignedDurationMs = 10000;
  world.emergencyLane = null;
  world.emergencyActive = false;
  world.emergencyCarSpawned = false;
  world.controlLane = null;
  world.controlLocked = false;
  world.finished = false;

  events.length = 0;
  eventLogNode.textContent = "Events will appear here...";
  renderMetrics({});
  renderCounts();
  updatePhaseText();
  drawScene();
  renderCycleMeta({});
}

async function runSyntheticCycle() {
  const body = {
    time_of_day: timeOfDayNode.value,
    intensity: Number(intensityNode.value),
    seed: Number(seedNode.value || 42),
    tick: syntheticTick,
    current_counts: world.counts,
    fairness_mode: fairnessModeNode.value,
  };

  const response = await fetch("/api/synthetic_cycle", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "Failed synthetic cycle");
  }

  renderMetrics(payload.congestion_metrics || {});
  renderFairness(payload.fairness || payload.model_outputs?.fairness || {});
  applySyntheticTick(payload, "synthetic_cycle");
  syntheticTick += 1;
}

async function spawnAmbulance() {
  const response = await fetch("/api/spawn_ambulance", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      current_counts: world.counts,
      lane: ambulanceLaneNode.value,
      fairness_mode: fairnessModeNode.value,
    }),
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "Failed to spawn ambulance");
  }

  addEvent(`ambulance_spawn: ${payload.scenario?.lane || "auto"}`);
  renderFairness(payload.fairness || payload.model_outputs?.fairness || {});
  applySyntheticTick(payload, "ambulance_spawn");
}

async function resetSyntheticRuntime() {
  const response = await fetch("/api/synthetic_reset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });
  if (!response.ok) {
    throw new Error("Failed to reset synthetic runtime");
  }
}

function stopAutoFlow() {
  if (autoFlowTimer) {
    window.clearInterval(autoFlowTimer);
    autoFlowTimer = 0;
  }
  autoFlowRunning = false;
  autoTickBusy = false;
  autoBtn.textContent = "Start Auto Flow";
}

function startAutoFlow() {
  if (autoFlowRunning) return;
  autoFlowRunning = true;
  autoBtn.textContent = "Stop Auto Flow";
  autoFlowTimer = window.setInterval(async () => {
    if (!autoFlowRunning || autoModeNode.value !== "on" || autoTickBusy) return;
    autoTickBusy = true;
    try {
      await runSyntheticCycle();
      setMessage("Auto flow tick completed.");
    } catch (error) {
      stopAutoFlow();
      setMessage(error.message || "Auto flow failed", true);
    } finally {
      autoTickBusy = false;
    }
  }, 3000);
}

intensityNode.addEventListener("input", () => {
  intensityValueNode.textContent = `${Number(intensityNode.value).toFixed(1)}x`;
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setMessage("Generating synthetic traffic...");
  stepBtn.disabled = true;

  runToken = Date.now();
  try {
    await runSyntheticCycle();
    setMessage("Synthetic cycle generated.");
  } catch (error) {
    setMessage(error.message || "Unexpected error", true);
  } finally {
    stepBtn.disabled = false;
  }
});

autoBtn.addEventListener("click", async () => {
  if (autoFlowRunning) {
    stopAutoFlow();
    setMessage("Auto flow stopped.");
    return;
  }

  if (autoModeNode.value !== "on") {
    setMessage("Set Continuous Flow to Auto every 3s first.", true);
    return;
  }

  setMessage("Starting auto flow...");
  try {
    await runSyntheticCycle();
    startAutoFlow();
    setMessage("Auto flow started.");
  } catch (error) {
    stopAutoFlow();
    setMessage(error.message || "Auto flow start failed", true);
  }
});

autoModeNode.addEventListener("change", () => {
  if (autoModeNode.value !== "on") {
    stopAutoFlow();
  }
});

spawnBtn.addEventListener("click", async () => {
  setMessage("Spawning ambulance and forcing corridor preemption...");
  spawnBtn.disabled = true;
  try {
    await spawnAmbulance();
    setMessage("Ambulance preemption applied.");
  } catch (error) {
    setMessage(error.message || "Unexpected error", true);
  } finally {
    spawnBtn.disabled = false;
  }
});

resetBtn.addEventListener("click", () => {
  stopAutoFlow();
  syntheticTick = 0;
  resetSyntheticRuntime()
    .catch(() => {
      setMessage("Reset partially completed (server state not cleared).", true);
    })
    .finally(() => {
      resetWorld();
      renderFairness({});
      emergencyBanner.classList.add("hidden");
      emergencyBanner.textContent = "";
      simStatus.textContent = "Ready for synthetic traffic generation.";
      setMessage("Synthetic demo reset.");
    });
});

resetWorld();
intensityValueNode.textContent = `${Number(intensityNode.value).toFixed(1)}x`;
renderFairness({});
