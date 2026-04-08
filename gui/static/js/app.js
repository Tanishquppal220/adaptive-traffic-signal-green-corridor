const laneKeys = ["laneN", "laneS", "laneE", "laneW"];
const laneLabels = {
  laneN: "North",
  laneS: "South",
  laneE: "East",
  laneW: "West",
};

const form = document.getElementById("laneUploadForm");
const runBtn = document.getElementById("runBtn");
const resetBtn = document.getElementById("resetBtn");
const formMessage = document.getElementById("formMessage");
const simStatus = document.getElementById("simStatus");
const emergencyBanner = document.getElementById("emergencyBanner");

const modeValue = document.getElementById("modeValue");
const directionValue = document.getElementById("directionValue");
const durationValue = document.getElementById("durationValue");
const actionValue = document.getElementById("actionValue");

const phaseLaneNode = document.getElementById("phaseLane");
const phaseSignalNode = document.getElementById("phaseSignal");
const phaseRemainingNode = document.getElementById("phaseRemaining");

const detectorTotalNode = document.getElementById("detectorTotal");
const detectorCountsNode = document.getElementById("detectorCounts");

const densityHorizonNode = document.getElementById("densityHorizon");
const densityValuesNode = document.getElementById("densityValues");

const emergencyDetectedNode = document.getElementById("emergencyDetected");
const emergencyLabelNode = document.getElementById("emergencyLabel");
const emergencyConfidenceNode = document.getElementById("emergencyConfidence");
const emergencyDirectionNode = document.getElementById("emergencyDirection");

const sirenDetectedNode = document.getElementById("sirenDetected");
const sirenConfidenceNode = document.getElementById("sirenConfidence");
const sirenSampleRateNode = document.getElementById("sirenSampleRate");

const dqnActionNode = document.getElementById("dqnAction");
const dqnDirectionNode = document.getElementById("dqnDirection");
const dqnDurationNode = document.getElementById("dqnDuration");

const canvas = document.getElementById("intersectionCanvas");
const ctx = canvas.getContext("2d");

let animationId = 0;
let runToken = 0;
let lastTs = 0;

const world = {
  counts: { laneN: 0, laneS: 0, laneE: 0, laneW: 0 },
  initialCounts: { laneN: 0, laneS: 0, laneE: 0, laneW: 0 },
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
  laneTimings: {},
  phaseCycles: { laneN: 0, laneS: 0, laneE: 0, laneW: 0 },
  elapsedMs: 0,
  baselineEstimateMs: 0,
  awaitingNext: false,
  waitingServer: false,
  autoNextPending: false,
  lastServedLane: null,
  pendingDecision: null,
  preemptionBufferMs: 0,
  emergencyLane: null,
  emergencyActive: false,
  emergencyCarSpawned: false,
  finished: false,
  seededRand: Math.random,
};

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
  formMessage.style.color = isError ? "#8f1b1b" : "#2b4a2f";
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
    const s = Math.max(0, Math.ceil(world.phaseTimeMs / 1000));
    phaseRemainingNode.textContent = `${s}s`;
  } else {
    phaseRemainingNode.textContent = "-";
  }
}

function renderTimingTable() {
  laneKeys.forEach((lane) => {
    const timingNode = document.getElementById(`timing-${lane}`);
    const cyclesNode = document.getElementById(`cycles-${lane}`);
    const duration = world.laneTimings[lane];
    if (timingNode) {
      timingNode.textContent = Number.isFinite(duration) ? String(duration) : "-";
    }
    if (cyclesNode) {
      cyclesNode.textContent = String(world.phaseCycles[lane] || 0);
    }

    const row = document.querySelector(`#timingTableBody tr[data-lane='${lane}']`);
    if (row) {
      row.classList.toggle("active", world.activeLane === lane && world.signal === "green");
    }
  });
}

function renderModelOutputs(modelOutputs = {}) {
  const detector = modelOutputs.detector || {};
  const density = modelOutputs.density || {};
  const emergency = modelOutputs.emergency || {};
  const siren = modelOutputs.siren || {};
  const dqn = modelOutputs.dqn || {};

  const dCounts = detector.lane_counts || {};
  detectorTotalNode.textContent = detector.total ?? "-";
  detectorCountsNode.textContent =
    `N:${dCounts.laneN ?? 0} S:${dCounts.laneS ?? 0} E:${dCounts.laneE ?? 0} W:${dCounts.laneW ?? 0}`;

  const p = density.predictions || {};
  densityHorizonNode.textContent = density.horizon_sec ? `${density.horizon_sec}s` : "-";
  densityValuesNode.textContent =
    `N:${Math.round(p.N || 0)} S:${Math.round(p.S || 0)} ` +
    `E:${Math.round(p.E || 0)} W:${Math.round(p.W || 0)}`;

  emergencyDetectedNode.textContent = emergency.detected ? "YES" : "NO";
  emergencyLabelNode.textContent = emergency.label || "-";
  emergencyConfidenceNode.textContent = Number(emergency.confidence || 0).toFixed(2);
  if (emergency.lane_counts) {
    const c = emergency.lane_counts;
    emergencyDirectionNode.textContent =
      `${emergency.direction || emergency.status || "-"} ` +
      `(N:${c.laneN ?? 0} S:${c.laneS ?? 0} E:${c.laneE ?? 0} W:${c.laneW ?? 0})`;
  } else {
    emergencyDirectionNode.textContent = emergency.direction || emergency.status || "-";
  }

  sirenDetectedNode.textContent = siren.detected ? "YES" : "NO";
  sirenConfidenceNode.textContent = Number(siren.confidence || 0).toFixed(2);
  sirenSampleRateNode.textContent = siren.sample_rate ? `${siren.sample_rate}Hz` : "-";

  dqnActionNode.textContent = dqn.action ?? "-";
  dqnDirectionNode.textContent = dqn.direction || "-";
  dqnDurationNode.textContent = dqn.duration ? `${dqn.duration}s` : "-";
  renderComparisonMetrics();
}

function renderComparisonMetrics() {
  const baselineNode = document.getElementById("baselineEstimate");
  const runtimeNode = document.getElementById("dqnRuntime");
  const speedupNode = document.getElementById("speedupLabel");

  if (baselineNode) {
    baselineNode.textContent = world.baselineEstimateMs > 0
      ? `${Math.round(world.baselineEstimateMs / 1000)}s`
      : "-";
  }
  if (runtimeNode) {
    runtimeNode.textContent = world.elapsedMs > 0
      ? `${Math.round(world.elapsedMs / 1000)}s`
      : "-";
  }
  if (speedupNode) {
    if (world.baselineEstimateMs > 0 && world.elapsedMs > 0) {
      const ratio = world.baselineEstimateMs / world.elapsedMs;
      if (ratio === 1) {
        speedupNode.textContent = "same as baseline";
      } else if (ratio > 1) {
        speedupNode.textContent = `${ratio.toFixed(2)}x faster`;
      } else {
        speedupNode.textContent = `${(1 / ratio).toFixed(2)}x slower`;
      }
    } else {
      speedupNode.textContent = "-";
    }
  }
}

function estimateBaselineTimeMs(
  counts,
  greenDurationSec,
  yellowMs,
  clearRateMin,
  clearRateMax
) {
  const laneOrder = ["laneN", "laneS", "laneE", "laneW"];
  const avgClearRate = (clearRateMin + clearRateMax) / 2;
  const greenMs = greenDurationSec * 1000;
  const cycleMs = greenMs + yellowMs;
  const queue = {
    laneN: counts.laneN || 0,
    laneS: counts.laneS || 0,
    laneE: counts.laneE || 0,
    laneW: counts.laneW || 0,
  };
  let total = 0;

  while (laneOrder.some((lane) => queue[lane] > 0)) {
    for (const lane of laneOrder) {
      if (!laneOrder.some((l) => queue[l] > 0)) {
        break;
      }
      const cars = queue[lane];
      if (cars > 0) {
        const removed = Math.min(cars, Math.round(avgClearRate * greenDurationSec));
        queue[lane] = Math.max(0, cars - removed);
      }
      total += cycleMs;
      if (!laneOrder.some((l) => queue[l] > 0)) {
        break;
      }
    }
  }

  return total;
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

function totalCars() {
  return laneKeys.reduce((acc, lane) => acc + (world.counts[lane] || 0), 0);
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

  ctx.strokeStyle = "#f9f3cc";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(g.cx - g.half, g.stopN);
  ctx.lineTo(g.cx + g.half, g.stopN);
  ctx.moveTo(g.cx - g.half, g.stopS);
  ctx.lineTo(g.cx + g.half, g.stopS);
  ctx.moveTo(g.stopW, g.cy - g.half);
  ctx.lineTo(g.stopW, g.cy + g.half);
  ctx.moveTo(g.stopE, g.cy - g.half);
  ctx.lineTo(g.stopE, g.cy + g.half);
  ctx.stroke();

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
    if (count > renderCount) {
      const pose = queueCarPose(lane, renderCount, g);
      ctx.fillStyle = "#fff";
      ctx.font = "12px IBM Plex Mono";
      ctx.fillText(`+${count - renderCount}`, pose.x + 12, pose.y + 3);
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
  world.phaseCycles[lane] = (world.phaseCycles[lane] || 0) + 1;
  world.lastServedLane = lane;
  world.awaitingNext = false;

  simStatus.textContent =
    `Green phase for ${laneLabels[lane]} (${Math.round(world.assignedDurationMs / 1000)}s)`;
  updatePhaseText();
  renderTimingTable();
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
  world.autoNextPending = false;
  simStatus.textContent = "Simulation finished: all lanes cleared.";
  setMessage("Simulation complete.");
  updatePhaseText();
  renderTimingTable();
}

function pauseForNextStep() {
  world.signal = "red";
  world.activeLane = null;
  world.phaseType = "paused";
  world.phaseTimeMs = 0;
  world.awaitingNext = true;

  if (totalCars() <= 0) {
    completeSimulation();
    return;
  }

  simStatus.textContent = "Step complete. Automatically running next decision...";
  world.autoNextPending = true;
  updatePhaseText();
}

function tickPhase(dtMs) {
  if (world.phaseType === "done" || world.phaseType === "idle" || world.phaseType === "paused") {
    updatePhaseText();
    renderCounts();
    renderComparisonMetrics();
    return;
  }

  if (world.signal === "green" && world.activeLane) {
    world.clearAccumulatorMs += dtMs;

    while (world.clearAccumulatorMs >= 1000) {
      world.clearAccumulatorMs -= 1000;
      const lane = world.activeLane;
      if ((world.counts[lane] || 0) > 0) {
        const remove = Math.min(
          world.counts[lane],
          randomInt(world.clearRateMin, world.clearRateMax)
        );
        world.counts[lane] -= remove;
        for (let i = 0; i < remove; i += 1) {
          const isEmergency =
            world.emergencyActive &&
            lane === world.emergencyLane &&
            !world.emergencyCarSpawned;
          if (isEmergency) {
            world.emergencyCarSpawned = true;
          }
          spawnMovingCar(lane, isEmergency);
        }
      }
    }
  }

  if (world.phaseType === "fixed-green") {
    world.phaseTimeMs -= dtMs;
    if (world.phaseTimeMs <= 0) {
      enterYellow();
    }
  } else if (world.phaseType === "yellow") {
    world.phaseTimeMs -= dtMs;
    if (world.phaseTimeMs <= 0) {
      if (world.pendingDecision) {
        const pending = { ...world.pendingDecision };
        world.pendingDecision = null;
        world.assignedDurationMs = pending.durationMs;
        enterGreen(pending.lane);
      } else {
        pauseForNextStep();
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
    if (emergencyBanner.textContent) {
      emergencyBanner.classList.add("hidden");
      emergencyBanner.textContent = "";
    }
  }
}

function animationLoop(ts) {
  if (runToken === 0) {
    return;
  }
  if (!lastTs) {
    lastTs = ts;
  }
  const dtMs = Math.min(50, ts - lastTs);
  lastTs = ts;
  if (!world.finished && runToken !== 0) {
    world.elapsedMs += dtMs;
  }

  tickPhase(dtMs);
  tickMovingCars(dtMs);
  drawScene();

  if (world.awaitingNext && !world.waitingServer && !world.finished && totalCars() > 0 && world.autoNextPending) {
    world.autoNextPending = false;
    requestNextCycle();
  }

  if (!world.finished || world.movingCars.length > 0) {
    animationId = window.requestAnimationFrame(animationLoop);
  }
}

function applyDecision(payload, initializeCounts = false) {
  const sim = payload.simulation;
  const decision = payload.decision;

  if (initializeCounts) {
    world.counts = { ...sim.initial_counts };
    world.initialCounts = { ...sim.initial_counts };
    world.movingCars = [];
    world.phaseCycles = { laneN: 0, laneS: 0, laneE: 0, laneW: 0 };
    world.finished = false;
  }

  world.clearRateMin = sim.clear_rate_min || 1;
  world.clearRateMax = sim.clear_rate_max || 2;
  world.yellowMs = sim.yellow_ms || 900;
  world.laneTimings = { ...(sim.lane_timings || {}) };
  world.seededRand = mulberry32(sim.seed || Date.now());
  world.preemptionBufferMs = (sim.preemption_buffer_sec || 0) * 1000;
  applyEmergencyVisualState(sim);
  if (initializeCounts) {
    world.elapsedMs = 0;
    world.baselineEstimateMs = estimateBaselineTimeMs(
      sim.initial_counts,
      30,
      world.yellowMs,
      world.clearRateMin,
      world.clearRateMax
    );
    renderComparisonMetrics();
  }

  modeValue.textContent = decision.mode || "-";
  directionValue.textContent = decision.direction || "-";
  durationValue.textContent = `${decision.duration || 0}s`;
  actionValue.textContent = decision.action ?? "-";

  renderModelOutputs(payload.model_outputs || {});
  renderCounts();

  if (
    world.preemptionBufferMs > 0 &&
    world.lastServedLane &&
    world.lastServedLane !== sim.selected_lane
  ) {
    world.pendingDecision = {
      lane: sim.selected_lane,
      durationMs: (sim.selected_duration || 1) * 1000,
    };
    world.assignedDurationMs = world.preemptionBufferMs;
    simStatus.textContent =
      `Emergency preemption buffer (${Math.round(world.preemptionBufferMs / 1000)}s) before switching.`;
    enterGreen(world.lastServedLane);
  } else {
    world.pendingDecision = null;
    world.assignedDurationMs = (sim.selected_duration || 1) * 1000;
    enterGreen(sim.selected_lane);
  }

  renderTimingTable();
  drawScene();
}

function startSimulation(payload) {
  applyDecision(payload, true);

  if (animationId) {
    window.cancelAnimationFrame(animationId);
  }
  lastTs = 0;
  animationId = window.requestAnimationFrame(animationLoop);
}

function resetWorld() {
  runToken = 0;
  if (animationId) {
    window.cancelAnimationFrame(animationId);
  }
  animationId = 0;
  lastTs = 0;

  world.counts = { laneN: 0, laneS: 0, laneE: 0, laneW: 0 };
  world.initialCounts = { ...world.counts };
  world.movingCars = [];
  world.sequence = [];
  world.sequenceCursor = 0;
  world.awaitingNext = false;
  world.waitingServer = false;
  world.autoNextPending = false;
  world.elapsedMs = 0;
  world.baselineEstimateMs = 0;
  world.lastServedLane = null;
  world.pendingDecision = null;
  world.preemptionBufferMs = 0;
  world.activeLane = null;
  world.signal = "red";
  world.phaseType = "idle";
  world.phaseTimeMs = 0;
  world.clearAccumulatorMs = 0;
  world.assignedDurationMs = 10000;
  world.laneTimings = {};
  world.phaseCycles = { laneN: 0, laneS: 0, laneE: 0, laneW: 0 };
  world.emergencyLane = null;
  world.emergencyActive = false;
  world.emergencyCarSpawned = false;
  world.finished = false;

  renderCounts();
  updatePhaseText();
  renderTimingTable();
  drawScene();
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setMessage("Processing images and running orchestrator...");
  simStatus.textContent = "Preparing simulation...";
  runBtn.disabled = true;

  runToken = Date.now();
  const localToken = runToken;
  const formData = new FormData(form);

  try {
    const response = await fetch("/api/run_cycle", {
      method: "POST",
      body: formData,
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Failed to run cycle");
    }

    if (localToken !== runToken) {
      return;
    }

    startSimulation(payload);
    setMessage("Cycle started. Animation is running.");
  } catch (error) {
    setMessage(error.message || "Unexpected error", true);
    simStatus.textContent = "Simulation could not start.";
  } finally {
    runBtn.disabled = false;
  }
});

async function requestNextCycle() {
  if (world.waitingServer || world.finished || !world.awaitingNext) {
    return;
  }

  world.autoNextPending = false;
  world.waitingServer = true;
  setMessage("Running next cycle inference...");
  simStatus.textContent = "Waiting for next-cycle decision...";

  try {
    const response = await fetch("/api/next_cycle", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        lane_counts: { ...world.counts },
        current_active_lane: world.lastServedLane,
      }),
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Failed to run next cycle");
    }

    applyDecision(payload, false);
    renderCounts();
    updatePhaseText();
    renderComparisonMetrics();
    setMessage("Next cycle started.");
  } catch (error) {
    setMessage(error.message || "Unexpected error", true);
    simStatus.textContent = "Could not run next cycle.";
  } finally {
    world.waitingServer = false;
  }
}

resetBtn.addEventListener("click", () => {
  form.reset();
  resetWorld();
  modeValue.textContent = "-";
  directionValue.textContent = "-";
  durationValue.textContent = "-";
  actionValue.textContent = "-";
  renderModelOutputs({});
  emergencyBanner.classList.add("hidden");
  emergencyBanner.textContent = "";
  simStatus.textContent = "Waiting for upload...";
  setMessage("Form reset.");
});

resetWorld();
renderModelOutputs({});
