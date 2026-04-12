const laneKeys = ["laneN", "laneS", "laneE", "laneW"];
const laneLabels = {
  laneN: "North",
  laneS: "South",
  laneE: "East",
  laneW: "West",
};

const byId = (id) => document.getElementById(id);

const form = byId("laneUploadForm");
const runBtn = byId("runBtn");
const resetBtn = byId("resetBtn");
const formMessage = byId("formMessage");
const simStatus = byId("simStatus");
const emergencyBanner = byId("emergencyBanner");

const backendStatusChip = byId("backendStatusChip");
const backendStatusValue = byId("backendStatusValue");
const controlModeChip = byId("controlModeChip");
const controlModeValue = byId("controlModeValue");
const runModeChip = byId("runModeChip");
const runModeValue = byId("runModeValue");

const phaseLaneNode = byId("phaseLane");
const phaseSignalNode = byId("phaseSignal");
const phaseRemainingNode = byId("phaseRemaining");

const detectorTotalNode = byId("detectorTotal");
const detectorCountsNode = byId("detectorCounts");

const densityHorizonNode = byId("densityHorizon");
const densityValuesNode = byId("densityValues");

const emergencyDetectedNode = byId("emergencyDetected");
const emergencyLabelNode = byId("emergencyLabel");
const emergencyConfidenceNode = byId("emergencyConfidence");
const emergencyDirectionNode = byId("emergencyDirection");

const sirenDetectedNode = byId("sirenDetected");
const sirenConfidenceNode = byId("sirenConfidence");
const sirenSampleRateNode = byId("sirenSampleRate");

const dqnActionNode = byId("dqnAction");
const dqnDirectionNode = byId("dqnDirection");
const dqnDurationNode = byId("dqnDuration");

const canvas = byId("intersectionCanvas");
const ctx = canvas ? canvas.getContext("2d") : null;

let animationId = 0;
let runToken = 0;
let lastTs = 0;

const world = {
  counts: { laneN: 0, laneS: 0, laneE: 0, laneW: 0 },
  initialCounts: { laneN: 0, laneS: 0, laneE: 0, laneW: 0 },
  movingCars: [],
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
  runMode: "idle",
  mockTieCursor: 0,
};

function setText(node, value, fallback = "-") {
  if (!node) return;
  if (value === null || value === undefined || value === "") {
    node.textContent = fallback;
  } else {
    node.textContent = String(value);
  }
}

function swapStatusClass(node, stateClass) {
  if (!node) return;
  node.classList.remove("is-ok", "is-warn", "is-error", "is-mock");
  node.classList.add(stateClass);
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function asFinite(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function fmtFixed(value, digits = 2, fallback = "-") {
  const n = asFinite(value);
  return n === null ? fallback : n.toFixed(digits);
}

function normalizeCounts(inputCounts = {}) {
  const out = {};
  laneKeys.forEach((lane) => {
    const value = Number(inputCounts[lane]);
    out[lane] = Number.isFinite(value) ? Math.max(0, Math.floor(value)) : 0;
  });
  return out;
}

function laneFromDirection(direction) {
  const text = String(direction || "").trim().toUpperCase();
  if (text === "N") return "laneN";
  if (text === "S") return "laneS";
  if (text === "E") return "laneE";
  if (text === "W") return "laneW";
  return "laneN";
}

function directionFromLane(lane) {
  return String(lane || "laneN").replace("lane", "");
}

function encodeMockAction(direction, duration) {
  const map = { N: 0, S: 1, E: 2, W: 3 };
  const dir = String(direction || "N").toUpperCase();
  const d = clamp(Math.round(Number(duration) || 5), 5, 60);
  return (map[dir] ?? 0) * 56 + (d - 5);
}

function mulberry32(seed) {
  let t = seed >>> 0;
  return function seeded() {
    t += 0x6d2b79f5;
    let x = Math.imul(t ^ (t >>> 15), 1 | t);
    x ^= x + Math.imul(x ^ (x >>> 7), 61 | x);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

function randomInt(min, max) {
  return Math.floor(world.seededRand() * (max - min + 1)) + min;
}

function setMessage(text, isError = false) {
  if (!formMessage) return;
  formMessage.textContent = text || "";
  formMessage.style.color = isError ? "#ef8f8f" : "#8cd1ff";
}

function setBackendStatus(text, stateClass) {
  setText(backendStatusValue, text, "Unknown");
  swapStatusClass(backendStatusChip, stateClass);
}

function setControlMode(text, stateClass = "is-warn") {
  setText(controlModeValue, text, "Unknown");
  swapStatusClass(controlModeChip, stateClass);
}

function setRunMode(mode) {
  world.runMode = mode;
  if (mode === "live") {
    setText(runModeValue, "Live Backend");
    swapStatusClass(runModeChip, "is-ok");
    return;
  }
  if (mode === "mock") {
    setText(runModeValue, "Simulated Fallback");
    swapStatusClass(runModeChip, "is-mock");
    return;
  }
  setText(runModeValue, "Idle");
  swapStatusClass(runModeChip, "is-warn");
}

function renderCounts() {
  laneKeys.forEach((lane) => {
    setText(byId(`count-${lane}`), Math.max(0, world.counts[lane] || 0), "0");
  });
}

function updatePhaseText() {
  setText(phaseLaneNode, world.activeLane ? laneLabels[world.activeLane] : "-", "-");
  setText(phaseSignalNode, String(world.signal || "RED").toUpperCase(), "RED");
  if (world.phaseType === "fixed-green" || world.phaseType === "yellow") {
    const s = Math.max(0, Math.ceil(world.phaseTimeMs / 1000));
    setText(phaseRemainingNode, `${s}s`);
    return;
  }
  setText(phaseRemainingNode, "-");
}

function renderTimingTable() {
  laneKeys.forEach((lane) => {
    const duration = world.laneTimings[lane];
    setText(byId(`timing-${lane}`), Number.isFinite(duration) ? duration : "-", "-");
    setText(byId(`cycles-${lane}`), world.phaseCycles[lane] || 0, "0");

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

  const simulated = world.runMode === "mock" || String(dqn.mode || "").includes("mock");

  const detectorCounts = detector.lane_counts || {};
  const nCount = detectorCounts.laneN ?? 0;
  const sCount = detectorCounts.laneS ?? 0;
  const eCount = detectorCounts.laneE ?? 0;
  const wCount = detectorCounts.laneW ?? 0;
  setText(detectorTotalNode, detector.total ?? "-", "-");
  setText(detectorCountsNode, `N:${nCount} S:${sCount} E:${eCount} W:${wCount}`, "-");

  const pred = density.predictions || {};
  setText(densityHorizonNode, density.horizon_sec ? `${density.horizon_sec}s` : "-", "-");
  setText(
    densityValuesNode,
    `N:${Math.round(asFinite(pred.N) ?? 0)} S:${Math.round(asFinite(pred.S) ?? 0)} ` +
      `E:${Math.round(asFinite(pred.E) ?? 0)} W:${Math.round(asFinite(pred.W) ?? 0)}`,
    "-"
  );

  if (simulated) {
    setText(emergencyDetectedNode, "NO (simulated)");
    setText(emergencyLabelNode, "Simulated");
    setText(emergencyConfidenceNode, "N/A");
    setText(emergencyDirectionNode, "N/A");
    setText(sirenDetectedNode, "NO (simulated)");
    setText(sirenConfidenceNode, "N/A");
    setText(sirenSampleRateNode, "N/A");
  } else {
    setText(emergencyDetectedNode, emergency.detected ? "YES" : "NO");
    setText(emergencyLabelNode, emergency.label || "-");
    setText(emergencyConfidenceNode, fmtFixed(emergency.confidence));
    if (emergency.lane_counts) {
      const c = emergency.lane_counts;
      setText(
        emergencyDirectionNode,
        `${emergency.direction || emergency.status || "-"} ` +
          `(N:${c.laneN ?? 0} S:${c.laneS ?? 0} E:${c.laneE ?? 0} W:${c.laneW ?? 0})`,
        "-"
      );
    } else {
      setText(emergencyDirectionNode, emergency.direction || emergency.status || "-", "-");
    }

    setText(sirenDetectedNode, siren.detected ? "YES" : "NO");
    setText(sirenConfidenceNode, fmtFixed(siren.confidence));
    setText(
      sirenSampleRateNode,
      asFinite(siren.sample_rate) !== null ? `${Math.round(asFinite(siren.sample_rate))}Hz` : "-",
      "-"
    );
  }

  setText(dqnActionNode, dqn.action ?? "-", "-");
  setText(dqnDirectionNode, dqn.direction || "-", "-");
  setText(dqnDurationNode, asFinite(dqn.duration) !== null ? `${Math.round(dqn.duration)}s` : "-", "-");
  renderComparisonMetrics();
}

function renderComparisonMetrics() {
  const baselineNode = byId("baselineEstimate");
  const runtimeNode = byId("dqnRuntime");
  const speedupNode = byId("speedupLabel");

  if (world.baselineEstimateMs > 0) {
    setText(baselineNode, `${Math.round(world.baselineEstimateMs / 1000)}s`);
  } else {
    setText(baselineNode, "-");
  }

  if (world.elapsedMs > 0) {
    setText(runtimeNode, `${Math.round(world.elapsedMs / 1000)}s`);
  } else {
    setText(runtimeNode, "-");
  }

  if (world.baselineEstimateMs > 0 && world.elapsedMs > 0) {
    const ratio = world.baselineEstimateMs / world.elapsedMs;
    if (Math.abs(ratio - 1) < 0.01) {
      setText(speedupNode, "same as baseline");
    } else if (ratio > 1) {
      setText(speedupNode, `${ratio.toFixed(2)}x faster`);
    } else {
      setText(speedupNode, `${(1 / ratio).toFixed(2)}x slower`);
    }
  } else {
    setText(speedupNode, "-");
  }
}

function estimateBaselineTimeMs(counts, greenDurationSec, yellowMs, clearRateMin, clearRateMax) {
  const laneOrder = ["laneN", "laneS", "laneE", "laneW"];
  const avgClearRate = (clearRateMin + clearRateMax) / 2;
  const greenMs = greenDurationSec * 1000;
  const cycleMs = greenMs + yellowMs;
  const queue = normalizeCounts(counts);
  let total = 0;

  while (laneOrder.some((lane) => queue[lane] > 0)) {
    for (const lane of laneOrder) {
      if (!laneOrder.some((l) => queue[l] > 0)) break;
      const cars = queue[lane];
      if (cars > 0) {
        const removed = Math.min(cars, Math.round(avgClearRate * greenDurationSec));
        queue[lane] = Math.max(0, cars - removed);
      }
      total += cycleMs;
      if (!laneOrder.some((l) => queue[l] > 0)) break;
    }
  }

  return total;
}

function applyEmergencyVisualState(sim) {
  if (!emergencyBanner) return;

  if (world.runMode === "mock") {
    emergencyBanner.classList.remove("hidden");
    emergencyBanner.textContent = "Simulated mode: emergency and siren outputs are fallback values.";
    return;
  }

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
function laneGeometry() {
  if (!canvas) {
    return null;
  }
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

function drawRoundedRect(x, y, w, h, r) {
  const radius = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.arcTo(x + w, y, x + w, y + h, radius);
  ctx.arcTo(x + w, y + h, x, y + h, radius);
  ctx.arcTo(x, y + h, x, y, radius);
  ctx.arcTo(x, y, x + w, y, radius);
  ctx.closePath();
}

function drawCar(x, y, angle, color, emergencyCar = false) {
  if (!ctx) return;
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(angle);

  ctx.fillStyle = color;
  drawRoundedRect(-12, -7, 24, 14, 4);
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
  if (!ctx) return;
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
  if (!ctx || !canvas) return;
  const g = laneGeometry();
  if (!g) return;

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

  setText(
    simStatus,
    `Green phase for ${laneLabels[lane]} (${Math.round(world.assignedDurationMs / 1000)}s)`
  );
  updatePhaseText();
  renderTimingTable();
}

function enterYellow() {
  world.signal = "yellow";
  world.phaseType = "yellow";
  world.phaseTimeMs = world.yellowMs;
  setText(simStatus, `Switching from ${laneLabels[world.activeLane]}...`);
  updatePhaseText();
}

function completeSimulation() {
  world.signal = "red";
  world.activeLane = null;
  world.phaseType = "done";
  world.finished = true;
  world.autoNextPending = false;
  setText(simStatus, "Simulation finished: all lanes cleared.");
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

  setText(simStatus, "Step complete. Automatically running next decision...");
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
        const remove = Math.min(world.counts[lane], randomInt(world.clearRateMin, world.clearRateMax));
        world.counts[lane] -= remove;
        for (let i = 0; i < remove; i += 1) {
          const isEmergency =
            world.emergencyActive && lane === world.emergencyLane && !world.emergencyCarSpawned;
          if (isEmergency) world.emergencyCarSpawned = true;
          spawnMovingCar(lane, isEmergency);
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
    if (emergencyBanner && emergencyBanner.textContent) {
      emergencyBanner.classList.add("hidden");
      emergencyBanner.textContent = "";
    }
  }
}

function animationLoop(ts) {
  if (runToken === 0) return;

  if (!lastTs) lastTs = ts;
  const dtMs = Math.min(50, ts - lastTs);
  lastTs = ts;

  if (!world.finished && runToken !== 0) world.elapsedMs += dtMs;

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
function pickMockLane(counts) {
  const queue = normalizeCounts(counts);
  const maxCount = Math.max(...laneKeys.map((lane) => queue[lane] || 0));
  const candidates = laneKeys.filter((lane) => (queue[lane] || 0) === maxCount);
  if (!candidates.length) return "laneN";

  for (let i = 0; i < laneKeys.length; i += 1) {
    const idx = (world.mockTieCursor + i) % laneKeys.length;
    const lane = laneKeys[idx];
    if (candidates.includes(lane)) {
      world.mockTieCursor = (idx + 1) % laneKeys.length;
      return lane;
    }
  }
  return candidates[0];
}

function computeMockDuration(queueForLane, maxQueue) {
  if (maxQueue <= 0 || queueForLane <= 0) return 5;
  const ratio = queueForLane / maxQueue;
  return clamp(Math.round(8 + ratio * 22), 5, 45);
}

function hashUploadFiles(formData) {
  let seed = 0x9e3779b9;
  laneKeys.concat(["sirenAudio"]).forEach((name, idx) => {
    const file = formData.get(name);
    const str = file && typeof file.name === "string" ? file.name : `${name}-none`;
    const size = file && Number.isFinite(file.size) ? file.size : 0;
    for (let i = 0; i < str.length; i += 1) {
      seed ^= str.charCodeAt(i) + idx + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    seed ^= size + idx * 17;
  });
  return seed >>> 0;
}

function buildMockLaneCountsFromUpload(formData) {
  const seed = hashUploadFiles(formData);
  const rand = mulberry32(seed || Date.now());
  const counts = { laneN: 0, laneS: 0, laneE: 0, laneW: 0 };

  laneKeys.forEach((lane) => {
    const file = formData.get(lane);
    const hasFile = file && typeof file.name === "string" && file.name.length > 0;
    if (!hasFile) {
      counts[lane] = 0;
      return;
    }
    counts[lane] = 4 + Math.floor(rand() * 20);
  });

  return counts;
}

function buildMockCyclePayload({ laneCounts, currentActiveLane = null, reason = "backend_unavailable" }) {
  const counts = normalizeCounts(laneCounts);
  const selectedLane = pickMockLane(counts);
  const selectedDirection = directionFromLane(selectedLane);
  const maxQueue = Math.max(...laneKeys.map((lane) => counts[lane] || 0));
  const selectedDuration = computeMockDuration(counts[selectedLane] || 0, maxQueue);

  const laneTimings = laneKeys.reduce((acc, lane) => {
    acc[lane] = lane === selectedLane ? selectedDuration : 0;
    return acc;
  }, {});

  const densityPred = {
    N: Math.round((counts.laneN || 0) * 1.08),
    S: Math.round((counts.laneS || 0) * 1.08),
    E: Math.round((counts.laneE || 0) * 1.08),
    W: Math.round((counts.laneW || 0) * 1.08),
  };

  const total = laneKeys.reduce((acc, lane) => acc + counts[lane], 0);
  const seed = Date.now() & 0xffffffff;
  const action = encodeMockAction(selectedDirection, selectedDuration);

  return {
    result: {
      lane_counts: { ...counts },
      direction: selectedDirection,
      duration: selectedDuration,
      mode: "mock-fallback",
      action,
      emergency: {
        detected: false,
        status: "simulated",
        label: null,
        confidence: 0,
        direction: null,
      },
      siren: {
        detected: false,
        confidence: 0,
        mode: "simulated",
        sample_rate: null,
      },
      detection: {
        mode: "simulated",
        total,
        lane_counts: { ...counts },
      },
      density: {
        mode: "simulated",
        horizon_sec: 60,
        predictions: { ...densityPred },
      },
    },
    simulation: {
      initial_counts: { ...counts },
      selected_lane: selectedLane,
      selected_direction: selectedDirection,
      selected_duration: selectedDuration,
      sequence: [selectedLane],
      lane_timings: laneTimings,
      decision_scope: "single_lane",
      active_lane_only_duration_applied: true,
      mode: "mock-fallback",
      emergency_detected: false,
      emergency_status: "simulated",
      emergency_release_reason: reason,
      emergency_message: "Simulated fallback mode active.",
      clear_rate_min: 1,
      clear_rate_max: 2,
      yellow_ms: 900,
      seed,
      tick_interval_sec: 3,
      preemption_buffer_sec: 0,
    },
    lane_counts: { ...counts },
    decision: {
      direction: selectedDirection,
      duration: selectedDuration,
      mode: "mock-fallback",
      action,
    },
    emergency: {
      detected: false,
      status: "simulated",
    },
    model_outputs: {
      detector: {
        mode: "simulated",
        total,
        lane_counts: { ...counts },
      },
      density: {
        mode: "simulated",
        horizon_sec: 60,
        predictions: { ...densityPred },
      },
      emergency: {
        detected: false,
        status: "simulated",
        label: null,
        confidence: null,
        direction: null,
        lane_counts: { laneN: 0, laneS: 0, laneE: 0, laneW: 0 },
        message: "Simulated fallback mode.",
      },
      siren: {
        detected: false,
        confidence: null,
        mode: "simulated",
        sample_rate: null,
      },
      dqn: {
        mode: "mock-fallback",
        action,
        direction: selectedDirection,
        duration: selectedDuration,
      },
    },
  };
}

function applyDecision(payload, initializeCounts = false) {
  const sim = payload?.simulation || {};
  const decision = payload?.decision || {};

  if (initializeCounts) {
    world.counts = normalizeCounts(sim.initial_counts || {});
    world.initialCounts = { ...world.counts };
    world.movingCars = [];
    world.phaseCycles = { laneN: 0, laneS: 0, laneE: 0, laneW: 0 };
    world.finished = false;
  }

  world.clearRateMin = clamp(Math.round(Number(sim.clear_rate_min) || 1), 1, 5);
  world.clearRateMax = clamp(Math.round(Number(sim.clear_rate_max) || 2), world.clearRateMin, 8);
  world.yellowMs = clamp(Math.round(Number(sim.yellow_ms) || 900), 300, 5000);
  world.laneTimings = { ...(sim.lane_timings || {}) };
  world.seededRand = mulberry32(sim.seed || Date.now());
  world.preemptionBufferMs = Math.max(0, Math.round(Number(sim.preemption_buffer_sec || 0) * 1000));

  if (initializeCounts) {
    world.elapsedMs = 0;
    world.baselineEstimateMs = estimateBaselineTimeMs(
      world.initialCounts,
      30,
      world.yellowMs,
      world.clearRateMin,
      world.clearRateMax
    );
    renderComparisonMetrics();
  }

  applyEmergencyVisualState(sim);
  renderModelOutputs(payload?.model_outputs || {});
  renderCounts();

  let selectedLane = sim.selected_lane || laneFromDirection(decision.direction);
  const selectedDurationSec = clamp(
    Math.round(Number(sim.selected_duration || decision.duration || 5)),
    1,
    60
  );

  const selectedQueue = world.counts[selectedLane] || 0;
  if (!initializeCounts && selectedQueue <= 0 && totalCars() > 0) {
    const fallbackLane = pickMockLane(world.counts);
    if (fallbackLane !== selectedLane) {
      selectedLane = fallbackLane;
      setText(
        simStatus,
        `Deadlock guard: switched to ${laneLabels[selectedLane]} (selected lane had zero queue).`
      );
    }
  }
  world.laneTimings = { laneN: 0, laneS: 0, laneE: 0, laneW: 0 };
  world.laneTimings[selectedLane] = selectedDurationSec;

  if (
    world.preemptionBufferMs > 0 &&
    world.lastServedLane &&
    world.lastServedLane !== selectedLane
  ) {
    world.pendingDecision = {
      lane: selectedLane,
      durationMs: selectedDurationSec * 1000,
    };
    world.assignedDurationMs = world.preemptionBufferMs;
    setText(
      simStatus,
      `Emergency preemption buffer (${Math.round(world.preemptionBufferMs / 1000)}s) before switching.`
    );
    enterGreen(world.lastServedLane);
  } else {
    world.pendingDecision = null;
    world.assignedDurationMs = selectedDurationSec * 1000;
    enterGreen(selectedLane);
  }

  renderTimingTable();
  drawScene();
}

function startSimulation(payload) {
  applyDecision(payload, true);
  if (animationId) window.cancelAnimationFrame(animationId);
  lastTs = 0;
  animationId = window.requestAnimationFrame(animationLoop);
}

function resetWorld() {
  runToken = 0;
  if (animationId) window.cancelAnimationFrame(animationId);
  animationId = 0;
  lastTs = 0;

  world.counts = { laneN: 0, laneS: 0, laneE: 0, laneW: 0 };
  world.initialCounts = { ...world.counts };
  world.movingCars = [];
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
  renderComparisonMetrics();
  drawScene();
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  let payload = null;
  try {
    payload = await response.json();
  } catch (_ignored) {
    payload = null;
  }
  if (!response.ok) {
    const message = payload?.error || `${response.status} ${response.statusText}`.trim();
    throw new Error(message || "Request failed");
  }
  return payload || {};
}

async function probeBackendStatus() {
  setBackendStatus("Checking...", "is-warn");
  setControlMode("Unknown", "is-warn");
  try {
    const status = await fetchJson("/api/status", { method: "GET" });
    setBackendStatus("Online", "is-ok");
    const mode = status?.mode ? String(status.mode) : "Unknown";
    setControlMode(mode.toUpperCase(), "is-ok");
  } catch (error) {
    setBackendStatus("Offline", "is-error");
    setControlMode("Unavailable", "is-warn");
    setMessage(`Backend status check failed: ${error.message}`, true);
  }
}
function refreshFileNameForInput(input) {
  if (!input || !input.name) return;
  const fileNameNode = byId(`fileName-${input.name}`);
  if (!fileNameNode) return;
  if (input.files && input.files.length > 0) {
    setText(fileNameNode, input.files[0].name, "No file selected");
  } else {
    setText(fileNameNode, "No file selected", "No file selected");
  }
}

function refreshAllFileNames() {
  if (!form) return;
  form.querySelectorAll("input[type='file'][name]").forEach((input) => {
    refreshFileNameForInput(input);
  });
}

function bindFileNameListeners() {
  if (!form) return;
  form.querySelectorAll("input[type='file'][name]").forEach((input) => {
    input.addEventListener("change", () => {
      refreshFileNameForInput(input);
    });
  });
}

function continueWithMock(reason, laneCounts) {
  setRunMode("mock");
  const payload = buildMockCyclePayload({
    laneCounts,
    currentActiveLane: world.lastServedLane,
    reason,
  });
  applyDecision(payload, false);
  renderCounts();
  updatePhaseText();
  renderComparisonMetrics();
}

async function requestNextCycle() {
  if (world.waitingServer || world.finished || !world.awaitingNext) return;

  world.autoNextPending = false;

  if (world.runMode === "mock") {
    continueWithMock("mock_mode_active", { ...world.counts });
    setMessage("Running simulated next-cycle inference.");
    return;
  }

  world.waitingServer = true;
  setMessage("Running next cycle inference...");
  setText(simStatus, "Waiting for next-cycle decision...");

  try {
    const payload = await fetchJson("/api/next_cycle", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        lane_counts: { ...world.counts },
        current_active_lane: world.lastServedLane,
      }),
    });

    applyDecision(payload, false);
    renderCounts();
    updatePhaseText();
    renderComparisonMetrics();
    setMessage("Next cycle started.");
  } catch (error) {
    setMessage(`Live next-cycle failed: ${error.message}. Switched to simulated fallback.`, true);
    continueWithMock("next_cycle_backend_failed", { ...world.counts });
    setText(simStatus, "Backend unavailable. Continuing in simulated mode.");
  } finally {
    world.waitingServer = false;
  }
}

if (form) {
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    setMessage("Processing images and running orchestrator...");
    setText(simStatus, "Preparing simulation...");
    if (runBtn) runBtn.disabled = true;

    runToken = Date.now();
    const localToken = runToken;
    const formData = new FormData(form);

    try {
      const payload = await fetchJson("/api/run_cycle", {
        method: "POST",
        body: formData,
      });

      if (localToken !== runToken) return;
      setRunMode("live");
      startSimulation(payload);
      setMessage("Live cycle started. Animation is running.");
    } catch (error) {
      if (localToken !== runToken) return;
      const mockCounts = buildMockLaneCountsFromUpload(formData);
      const payload = buildMockCyclePayload({
        laneCounts: mockCounts,
        reason: "run_cycle_backend_failed",
      });
      setRunMode("mock");
      startSimulation(payload);
      setMessage(`Live backend failed: ${error.message}. Running simulated fallback.`, true);
      setText(simStatus, "Simulated fallback running.");
    } finally {
      if (runBtn) runBtn.disabled = false;
    }
  });
}

if (resetBtn) {
  resetBtn.addEventListener("click", () => {
    if (form) form.reset();
    refreshAllFileNames();
    resetWorld();
    setRunMode("idle");
    renderModelOutputs({});
    if (emergencyBanner) {
      emergencyBanner.classList.add("hidden");
      emergencyBanner.textContent = "";
    }
    setText(simStatus, "Waiting for upload...");
    setMessage("Form reset.");
  });
}

setRunMode("idle");
resetWorld();
renderModelOutputs({});
bindFileNameListeners();
refreshAllFileNames();
probeBackendStatus();
