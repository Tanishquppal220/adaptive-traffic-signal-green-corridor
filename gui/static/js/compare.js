/**
 * compare.js — DQN vs Round-Robin side-by-side comparison engine
 *
 * Runs two independent simulation worlds on separate canvases.
 * DQN world: driven by /api/run_cycle + /api/next_cycle backend calls.
 * Round-Robin world: pure client-side fixed 15s rotation N→S→E→W.
 */

// ─────────────────────────────────────────────────────────────
// Constants & helpers
// ─────────────────────────────────────────────────────────────
const LANE_KEYS = ["laneN", "laneS", "laneE", "laneW"];
const LANE_LABELS = { laneN: "North", laneS: "South", laneE: "East", laneW: "West" };
const RR_DURATION_SEC = 15; // fixed Round-Robin green time per lane
const RR_SEQUENCE = ["laneN", "laneS", "laneE", "laneW"];
const byId = (id) => document.getElementById(id);

function clamp(v, lo, hi) { return Math.min(hi, Math.max(lo, v)); }
function asFinite(v) { const n = Number(v); return Number.isFinite(n) ? n : null; }
function setText(node, val, fb = "-") {
  if (!node) return;
  node.textContent = (val === null || val === undefined || val === "") ? fb : String(val);
}
function normalizeCounts(raw = {}) {
  const out = {};
  LANE_KEYS.forEach((l) => { const v = Number(raw[l]); out[l] = Number.isFinite(v) ? Math.max(0, Math.floor(v)) : 0; });
  return out;
}
function totalCars(counts) { return LANE_KEYS.reduce((a, l) => a + (counts[l] || 0), 0); }
function mulberry32(seed) {
  let t = seed >>> 0;
  return () => { t += 0x6d2b79f5; let x = Math.imul(t ^ (t >>> 15), 1 | t); x ^= x + Math.imul(x ^ (x >>> 7), 61 | x); return ((x ^ (x >>> 14)) >>> 0) / 4294967296; };
}
function laneFromDir(d) {
  const m = { N: "laneN", S: "laneS", E: "laneE", W: "laneW" };
  return m[String(d || "N").toUpperCase()] || "laneN";
}
function maxLane(counts) {
  let best = "laneN", max = -1;
  LANE_KEYS.forEach((l) => { if ((counts[l] || 0) > max) { max = counts[l]; best = l; } });
  return best;
}
function encodeMockAction(direction, duration) {
  const m = { N: 0, S: 1, E: 2, W: 3 };
  return (m[String(direction).toUpperCase()] ?? 0) * 56 + (clamp(Math.round(duration), 5, 60) - 5);
}
function hashCounts(counts) {
  return LANE_KEYS.reduce((s, l, i) => s ^ ((counts[l] || 0) * 31 + i * 7919), 0x9e3779b9) >>> 0;
}

// ─────────────────────────────────────────────────────────────
// World factory – creates an independent simulation state
// ─────────────────────────────────────────────────────────────
function createWorld() {
  return {
    counts: { laneN: 0, laneS: 0, laneE: 0, laneW: 0 },
    initialCounts: { laneN: 0, laneS: 0, laneE: 0, laneW: 0 },
    totalInitial: 0,
    movingCars: [],
    activeLane: null,
    signal: "red",
    phaseType: "idle",        // idle | fixed-green | yellow | paused | done
    phaseTimeMs: 0,
    clearAccumulatorMs: 0,
    clearRateMin: 1,
    clearRateMax: 2,
    yellowMs: 900,
    assignedDurationMs: 10000,
    phaseCycles: { laneN: 0, laneS: 0, laneE: 0, laneW: 0 },
    totalCycles: 0,
    elapsedMs: 0,
    carsCleared: 0,
    awaitingNext: false,
    waitingServer: false,
    autoNextPending: false,
    lastServedLane: null,
    finished: false,
    seededRand: Math.random,
    runMode: "idle",           // idle | live | mock
    // Round-Robin specific
    rrCursor: 0,
  };
}

// ─────────────────────────────────────────────────────────────
// Canvas / drawing helpers
// ─────────────────────────────────────────────────────────────
function laneGeometry(canvas) {
  const w = canvas.width, h = canvas.height;
  const cx = w / 2, cy = h / 2;
  const roadW = Math.min(w, h) * 0.38;
  const half = roadW / 2;
  return {
    w, h, cx, cy, roadW, half,
    xN: cx - 18, xS: cx + 18, yE: cy - 18, yW: cy + 18,
    stopN: cy - half - 8, stopS: cy + half + 8,
    stopE: cx + half + 8, stopW: cx - half - 8,
  };
}

function queueCarPose(lane, idx, g) {
  const sp = 28;
  if (lane === "laneN") return { x: g.xN, y: g.stopN - 22 - idx * sp, a: Math.PI / 2 };
  if (lane === "laneS") return { x: g.xS, y: g.stopS + 22 + idx * sp, a: -Math.PI / 2 };
  if (lane === "laneE") return { x: g.stopE + 22 + idx * sp, y: g.yE, a: Math.PI };
  return { x: g.stopW - 22 - idx * sp, y: g.yW, a: 0 };
}

function movingPath(lane, g) {
  if (lane === "laneN") return { sx: g.xN, sy: g.stopN - 6, ex: g.xN, ey: g.h + 40, a: Math.PI / 2 };
  if (lane === "laneS") return { sx: g.xS, sy: g.stopS + 6, ex: g.xS, ey: -40, a: -Math.PI / 2 };
  if (lane === "laneE") return { sx: g.stopE + 6, sy: g.yE, ex: -40, ey: g.yE, a: Math.PI };
  return { sx: g.stopW - 6, sy: g.yW, ex: g.w + 40, ey: g.yW, a: 0 };
}

function drawRoundedRect(ctx, x, y, w, h, r) {
  const rad = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + rad, y);
  ctx.arcTo(x + w, y, x + w, y + h, rad);
  ctx.arcTo(x + w, y + h, x, y + h, rad);
  ctx.arcTo(x, y + h, x, y, rad);
  ctx.arcTo(x, y, x + w, y, rad);
  ctx.closePath();
}

function drawCar(ctx, x, y, angle, color) {
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(angle);
  ctx.fillStyle = color;
  drawRoundedRect(ctx, -12, -7, 24, 14, 4);
  ctx.fill();
  ctx.fillStyle = "rgba(255,255,255,0.62)";
  ctx.fillRect(-7, -5, 14, 3);
  ctx.restore();
}

function drawSignals(ctx, g, world) {
  const signals = {
    laneN: { x: g.xN + 30, y: g.stopN + 16 },
    laneS: { x: g.xS - 30, y: g.stopS - 16 },
    laneE: { x: g.stopE - 16, y: g.yE - 30 },
    laneW: { x: g.stopW + 16, y: g.yW + 30 },
  };
  LANE_KEYS.forEach((lane) => {
    const p = signals[lane];
    const isActive = world.activeLane === lane;
    ctx.fillStyle = "#111";
    ctx.fillRect(p.x - 8, p.y - 22, 16, 42);
    const lamps = [
      { y: -14, c: "#d64545", on: !isActive || world.signal === "red" },
      { y: -2,  c: "#d9a404", on: isActive && world.signal === "yellow" },
      { y: 10,  c: "#2f9e44", on: isActive && world.signal === "green" },
    ];
    lamps.forEach((lamp) => {
      ctx.beginPath();
      ctx.arc(p.x, p.y + lamp.y, 4.2, 0, Math.PI * 2);
      ctx.fillStyle = lamp.on ? lamp.c : "#3d3d3d";
      ctx.fill();
    });
  });
}

function drawScene(canvas, ctx, world, accentColor) {
  if (!ctx || !canvas) return;
  const g = laneGeometry(canvas);
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Road surface
  ctx.fillStyle = "#2d343a";
  ctx.fillRect(0, 0, g.w, g.h);
  ctx.fillStyle = "#40484f";
  ctx.fillRect(g.cx - g.half, 0, g.roadW, g.h);
  ctx.fillRect(0, g.cy - g.half, g.w, g.roadW);

  // Centre dashes
  ctx.strokeStyle = "rgba(240,240,240,0.65)";
  ctx.lineWidth = 2;
  ctx.setLineDash([10, 10]);
  ctx.beginPath();
  ctx.moveTo(g.cx, 0); ctx.lineTo(g.cx, g.h);
  ctx.moveTo(0, g.cy); ctx.lineTo(g.w, g.cy);
  ctx.stroke();
  ctx.setLineDash([]);

  // Stop lines
  ctx.strokeStyle = "#f9f3cc";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(g.cx - g.half, g.stopN); ctx.lineTo(g.cx + g.half, g.stopN);
  ctx.moveTo(g.cx - g.half, g.stopS); ctx.lineTo(g.cx + g.half, g.stopS);
  ctx.moveTo(g.stopW, g.cy - g.half); ctx.lineTo(g.stopW, g.cy + g.half);
  ctx.moveTo(g.stopE, g.cy - g.half); ctx.lineTo(g.stopE, g.cy + g.half);
  ctx.stroke();

  // Active lane highlight
  if (world.activeLane && world.signal === "green") {
    ctx.fillStyle = accentColor + "33";
    ctx.fillRect(g.cx - g.half, 0, g.roadW, g.h);
    ctx.fillRect(0, g.cy - g.half, g.w, g.roadW);
  }

  // Queued cars
  const palette = ["#2a9d8f", "#2b8bd3", "#915bb8", "#c77d2d", "#3c5fcb"];
  LANE_KEYS.forEach((lane) => {
    const count = world.counts[lane] || 0;
    const render = Math.min(count, 18);
    for (let i = 0; i < render; i++) {
      const pose = queueCarPose(lane, i, g);
      drawCar(ctx, pose.x, pose.y, pose.a, "#4a90d9");
    }
    if (count > render) {
      const pose = queueCarPose(lane, render, g);
      ctx.fillStyle = "#fff";
      ctx.font = "12px IBM Plex Mono";
      ctx.fillText(`+${count - render}`, pose.x + 12, pose.y + 3);
    }
  });

  // Moving cars
  world.movingCars.forEach((car) => {
    const path = movingPath(car.lane, g);
    const x = path.sx + (path.ex - path.sx) * car.t;
    const y = path.sy + (path.ey - path.sy) * car.t;
    drawCar(ctx, x, y, path.a, car.color);
  });

  drawSignals(ctx, g, world);
}

// ─────────────────────────────────────────────────────────────
// Simulation phase logic (shared by both worlds)
// ─────────────────────────────────────────────────────────────
function spawnMovingCar(world, lane) {
  const palette = ["#2a9d8f", "#2b8bd3", "#915bb8", "#c77d2d", "#3c5fcb"];
  world.movingCars.push({
    lane, t: 0,
    speed: 0.35 + world.seededRand() * 0.25,
    color: palette[Math.floor(world.seededRand() * palette.length)],
  });
}

function enterGreen(world, lane, durationSec) {
  world.activeLane = lane;
  world.signal = "green";
  world.phaseType = "fixed-green";
  world.phaseTimeMs = durationSec * 1000;
  world.clearAccumulatorMs = 0;
  world.phaseCycles[lane] = (world.phaseCycles[lane] || 0) + 1;
  world.totalCycles++;
  world.lastServedLane = lane;
  world.awaitingNext = false;
}

function enterYellow(world) {
  world.signal = "yellow";
  world.phaseType = "yellow";
  world.phaseTimeMs = world.yellowMs;
}

function completeSimulation(world) {
  world.signal = "red";
  world.activeLane = null;
  world.phaseType = "done";
  world.finished = true;
  world.awaitingNext = false;
  world.autoNextPending = false;
}

function pauseForNext(world) {
  world.signal = "red";
  world.activeLane = null;
  world.phaseType = "paused";
  world.awaitingNext = true;
  if (totalCars(world.counts) <= 0) { completeSimulation(world); return; }
  world.autoNextPending = true;
}

function tickPhase(world, dtMs) {
  if (["done", "idle", "paused"].includes(world.phaseType)) return;

  if (world.signal === "green" && world.activeLane) {
    world.clearAccumulatorMs += dtMs;
    while (world.clearAccumulatorMs >= 1000) {
      world.clearAccumulatorMs -= 1000;
      const lane = world.activeLane;
      if ((world.counts[lane] || 0) > 0) {
        const remove = Math.min(world.counts[lane], 1 + Math.floor(world.seededRand() * 2));
        world.counts[lane] -= remove;
        world.carsCleared += remove;
        for (let i = 0; i < remove; i++) spawnMovingCar(world, lane);
      }
    }
  }

  if (world.phaseType === "fixed-green") {
    world.phaseTimeMs -= dtMs;
    if (world.phaseTimeMs <= 0) enterYellow(world);
  } else if (world.phaseType === "yellow") {
    world.phaseTimeMs -= dtMs;
    if (world.phaseTimeMs <= 0) pauseForNext(world);
  }
}

function tickMovingCars(world, dtMs) {
  world.movingCars.forEach((car) => { car.t += (car.speed * dtMs) / 1000; });
  world.movingCars = world.movingCars.filter((car) => car.t < 1.05);
}

// ─────────────────────────────────────────────────────────────
// Round-Robin — next lane decision (pure client-side)
// ─────────────────────────────────────────────────────────────
function rrNextLane(world) {
  // Advance cursor until we find a lane with cars, or exhaust all
  let attempts = 0;
  while (attempts < LANE_KEYS.length) {
    const lane = RR_SEQUENCE[world.rrCursor % RR_SEQUENCE.length];
    world.rrCursor++;
    if ((world.counts[lane] || 0) > 0) return lane;
    attempts++;
  }
  // All empty — just take next in sequence
  const lane = RR_SEQUENCE[world.rrCursor % RR_SEQUENCE.length];
  world.rrCursor++;
  return lane;
}

function applyRRDecision(world) {
  if (world.finished || totalCars(world.counts) <= 0) { completeSimulation(world); return; }
  const lane = rrNextLane(world);
  enterGreen(world, lane, RR_DURATION_SEC);
}

// ─────────────────────────────────────────────────────────────
// Mock DQN fallback (max-queue heuristic)
// ─────────────────────────────────────────────────────────────
function mockDQNDecision(world) {
  if (world.finished || totalCars(world.counts) <= 0) { completeSimulation(world); return; }
  const lane = maxLane(world.counts);
  const maxQ = Math.max(...LANE_KEYS.map((l) => world.counts[l] || 0));
  const ratio = maxQ > 0 ? (world.counts[lane] || 0) / maxQ : 0;
  const duration = clamp(Math.round(8 + ratio * 22), 5, 45);
  enterGreen(world, lane, duration);
}

// ─────────────────────────────────────────────────────────────
// UI update helpers
// ─────────────────────────────────────────────────────────────
function updateSimUI(prefix, world, statusEl) {
  LANE_KEYS.forEach((lane) => {
    setText(byId(`${prefix}-count-${lane}`), Math.max(0, world.counts[lane] || 0), "0");
  });
  setText(byId(`${prefix}-phaseLane`), world.activeLane ? LANE_LABELS[world.activeLane] : "-");
  setText(byId(`${prefix}-phaseSignal`), String(world.signal || "RED").toUpperCase());
  if (world.phaseType === "fixed-green" || world.phaseType === "yellow") {
    setText(byId(`${prefix}-phaseRemaining`), `${Math.max(0, Math.ceil(world.phaseTimeMs / 1000))}s`);
  } else {
    setText(byId(`${prefix}-phaseRemaining`), "-");
  }
  if (statusEl) {
    if (world.finished) {
      statusEl.textContent = "✓ Finished";
      statusEl.classList.add("pill-done");
    } else if (world.phaseType === "idle" || world.phaseType === "paused") {
      statusEl.textContent = "⏸ Waiting...";
    } else if (world.signal === "green") {
      statusEl.textContent = `🟢 Green — ${LANE_LABELS[world.activeLane]}`;
    } else if (world.signal === "yellow") {
      statusEl.textContent = `🟡 Yellow`;
    } else {
      statusEl.textContent = "🔴 Red";
    }
  }
}

function updateDQNDecisionUI(direction, duration, mode) {
  setText(byId("dqn-direction"), direction || "-");
  setText(byId("dqn-duration"), asFinite(duration) !== null ? `${Math.round(duration)}s` : "-");
  setText(byId("dqn-mode"), mode || "-");
}

// ─────────────────────────────────────────────────────────────
// Scoreboard & chart
// ─────────────────────────────────────────────────────────────
let chartInstance = null;
const chartTimeLabels = [];
const chartDqnData = [];
const chartRrData = [];
let chartTick = 0;
const CHART_INTERVAL_MS = 1500;
let lastChartUpdateMs = 0;

function initChart() {
  const canvas = byId("queueChart");
  if (!canvas || !window.Chart) return;
  if (chartInstance) { chartInstance.destroy(); chartInstance = null; }
  chartTimeLabels.length = 0;
  chartDqnData.length = 0;
  chartRrData.length = 0;
  chartTick = 0;
  lastChartUpdateMs = 0;

  chartInstance = new Chart(canvas, {
    type: "line",
    data: {
      labels: chartTimeLabels,
      datasets: [
        {
          label: "DQN — Total Queue",
          data: chartDqnData,
          borderColor: "#ff8d3a",
          backgroundColor: "rgba(255,141,58,0.15)",
          tension: 0.4,
          fill: true,
          pointRadius: 2,
        },
        {
          label: "Round-Robin — Total Queue",
          data: chartRrData,
          borderColor: "#4dc0ff",
          backgroundColor: "rgba(77,192,255,0.12)",
          tension: 0.4,
          fill: true,
          pointRadius: 2,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 300 },
      plugins: {
        legend: { labels: { color: "#edf3fa", font: { family: "Space Grotesk" } } },
      },
      scales: {
        x: {
          ticks: { color: "#9ab0c5", maxTicksLimit: 12 },
          grid: { color: "rgba(126,162,190,0.15)" },
          title: { display: true, text: "Time (s)", color: "#9ab0c5" },
        },
        y: {
          ticks: { color: "#9ab0c5" },
          grid: { color: "rgba(126,162,190,0.15)" },
          beginAtZero: true,
          title: { display: true, text: "Cars in Queue", color: "#9ab0c5" },
        },
      },
    },
  });
}

function maybeUpdateChart(nowMs, dqnW, rrW) {
  if (nowMs - lastChartUpdateMs < CHART_INTERVAL_MS) return;
  lastChartUpdateMs = nowMs;
  chartTick++;
  const t = Math.round((nowMs / 1000));
  chartTimeLabels.push(`${t}s`);
  chartDqnData.push(totalCars(dqnW.counts));
  chartRrData.push(totalCars(rrW.counts));
  if (chartTimeLabels.length > 60) {
    chartTimeLabels.shift();
    chartDqnData.shift();
    chartRrData.shift();
  }
  if (chartInstance) chartInstance.update("none");
}

function updateScoreboard(dqnW, rrW) {
  const toSec = (ms) => `${Math.round(ms / 1000)}s`;
  const throughput = (w) => {
    const sec = w.elapsedMs / 1000;
    return sec > 0 ? `${(w.carsCleared / sec).toFixed(2)}/s` : "-";
  };

  setText(byId("dqn-elapsed"), toSec(dqnW.elapsedMs));
  setText(byId("dqn-cleared"), dqnW.carsCleared);
  setText(byId("dqn-cycles"), dqnW.totalCycles);
  setText(byId("dqn-queue"), totalCars(dqnW.counts));
  setText(byId("dqn-throughput"), throughput(dqnW));

  setText(byId("rr-elapsed"), toSec(rrW.elapsedMs));
  setText(byId("rr-cleared"), rrW.carsCleared);
  setText(byId("rr-cycles"), rrW.totalCycles);
  setText(byId("rr-queue"), totalCars(rrW.counts));
  setText(byId("rr-throughput"), throughput(rrW));

  // Efficiency bar — proportion of total cars cleared
  const totalInit = Math.max(1, dqnW.totalInitial || rrW.totalInitial || 1);
  const dqnPct = clamp((dqnW.carsCleared / totalInit) * 100, 0, 100);
  const rrPct = clamp((rrW.carsCleared / totalInit) * 100, 0, 100);
  const dqnBar = byId("effBarDqn");
  const rrBar = byId("effBarRr");
  if (dqnBar) dqnBar.style.width = `${dqnPct}%`;
  if (rrBar) rrBar.style.width = `${rrPct}%`;
  setText(byId("effLabelDqn"), `DQN ${Math.round(dqnPct)}%`);
  setText(byId("effLabelRr"), `RR ${Math.round(rrPct)}%`);
}

function showWinner(dqnW, rrW) {
  const dqnDone = dqnW.finished;
  const rrDone = rrW.finished;
  const dqnBadge = byId("dqnWinnerBadge");
  const rrBadge = byId("rrWinnerBadge");
  if (!dqnDone && !rrDone) return;
  if (dqnDone && rrDone) {
    // Both done — compare elapsed
    if (!dqnBadge.classList.contains("shown") && !rrBadge.classList.contains("shown")) {
      if (dqnW.elapsedMs <= rrW.elapsedMs) {
        dqnBadge.classList.remove("hidden");
        dqnBadge.classList.add("shown");
      } else {
        rrBadge.classList.remove("hidden");
        rrBadge.classList.add("shown");
      }
    }
  } else if (dqnDone && !rrBadge.classList.contains("shown")) {
    dqnBadge.classList.remove("hidden");
    dqnBadge.classList.add("shown");
  } else if (rrDone && !dqnBadge.classList.contains("shown")) {
    rrBadge.classList.remove("hidden");
    rrBadge.classList.add("shown");
  }
}

// ─────────────────────────────────────────────────────────────
// Global state
// ─────────────────────────────────────────────────────────────
const dqnWorld = createWorld();
const rrWorld = createWorld();

const dqnCanvas = byId("dqnCanvas");
const rrCanvas = byId("rrCanvas");
const dqnCtx = dqnCanvas ? dqnCanvas.getContext("2d") : null;
const rrCtx = rrCanvas ? rrCanvas.getContext("2d") : null;

let animId = 0;
let lastTs = 0;
let runToken = 0;

// ─────────────────────────────────────────────────────────────
// Backend / fetch helpers
// ─────────────────────────────────────────────────────────────
async function fetchJson(url, options = {}) {
  const res = await fetch(url, options);
  let data = null;
  try { data = await res.json(); } catch (_) { data = null; }
  if (!res.ok) throw new Error(data?.error || `${res.status} ${res.statusText}`);
  return data || {};
}

async function callRunCycle(formData) {
  return fetchJson("/api/run_cycle", { method: "POST", body: formData });
}

async function callNextCycle(laneCounts, currentActiveLane) {
  return fetchJson("/api/next_cycle", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ lane_counts: laneCounts, current_active_lane: currentActiveLane }),
  });
}

// ─────────────────────────────────────────────────────────────
// Apply a backend payload to the DQN world
// ─────────────────────────────────────────────────────────────
function applyDQNPayload(payload, isFirst) {
  const sim = payload?.simulation || {};
  const decision = payload?.decision || {};

  if (isFirst) {
    dqnWorld.counts = normalizeCounts(sim.initial_counts || {});
    dqnWorld.initialCounts = { ...dqnWorld.counts };
    dqnWorld.totalInitial = totalCars(dqnWorld.counts);
    dqnWorld.movingCars = [];
    dqnWorld.phaseCycles = { laneN: 0, laneS: 0, laneE: 0, laneW: 0 };
    dqnWorld.totalCycles = 0;
    dqnWorld.carsCleared = 0;
    dqnWorld.finished = false;
    dqnWorld.elapsedMs = 0;
  }

  const seed = sim.seed || Date.now();
  dqnWorld.seededRand = mulberry32(seed);
  dqnWorld.clearRateMin = clamp(Math.round(Number(sim.clear_rate_min) || 1), 1, 5);
  dqnWorld.clearRateMax = clamp(Math.round(Number(sim.clear_rate_max) || 2), dqnWorld.clearRateMin, 8);
  dqnWorld.yellowMs = clamp(Math.round(Number(sim.yellow_ms) || 900), 300, 5000);

  const dir = decision.direction || sim.selected_direction || "N";
  const durSec = clamp(Math.round(Number(decision.duration || sim.selected_duration || 10)), 1, 60);
  const lane = laneFromDir(dir);

  updateDQNDecisionUI(dir, durSec, decision.mode || sim.mode || "dqn");

  // Guard: skip if selected lane is empty and others have cars
  let finalLane = lane;
  if (!isFirst && (dqnWorld.counts[lane] || 0) <= 0 && totalCars(dqnWorld.counts) > 0) {
    finalLane = maxLane(dqnWorld.counts);
  }

  enterGreen(dqnWorld, finalLane, durSec);
}

// ─────────────────────────────────────────────────────────────
// DQN next-cycle request
// ─────────────────────────────────────────────────────────────
async function dqnRequestNext(localToken) {
  if (dqnWorld.waitingServer || dqnWorld.finished) return;
  dqnWorld.waitingServer = true;
  dqnWorld.autoNextPending = false;

  if (dqnWorld.runMode === "mock") {
    dqnWorld.waitingServer = false;
    mockDQNDecision(dqnWorld);
    return;
  }

  try {
    const payload = await callNextCycle({ ...dqnWorld.counts }, dqnWorld.lastServedLane);
    if (localToken !== runToken) return;
    applyDQNPayload(payload, false);
  } catch (_err) {
    if (localToken !== runToken) return;
    dqnWorld.runMode = "mock";
    mockDQNDecision(dqnWorld);
  } finally {
    dqnWorld.waitingServer = false;
  }
}

// ─────────────────────────────────────────────────────────────
// Round-Robin next-cycle trigger
// ─────────────────────────────────────────────────────────────
function rrTriggerNext() {
  if (rrWorld.waitingServer || rrWorld.finished) return;
  rrWorld.autoNextPending = false;
  applyRRDecision(rrWorld);
}

// ─────────────────────────────────────────────────────────────
// Animation loop
// ─────────────────────────────────────────────────────────────
function loop(ts) {
  if (runToken === 0) return;
  if (!lastTs) lastTs = ts;
  const dt = Math.min(50, ts - lastTs);
  lastTs = ts;

  // Tick DQN world
  if (!dqnWorld.finished) dqnWorld.elapsedMs += dt;
  tickPhase(dqnWorld, dt);
  tickMovingCars(dqnWorld, dt);

  // Tick RR world
  if (!rrWorld.finished) rrWorld.elapsedMs += dt;
  tickPhase(rrWorld, dt);
  tickMovingCars(rrWorld, dt);

  // Draw
  drawScene(dqnCanvas, dqnCtx, dqnWorld, "#ff8d3a");
  drawScene(rrCanvas, rrCtx, rrWorld, "#4dc0ff");

  // Update UI
  updateSimUI("dqn", dqnWorld, byId("dqnStatus"));
  updateSimUI("rr", rrWorld, byId("rrStatus"));
  updateScoreboard(dqnWorld, rrWorld);
  maybeUpdateChart(dqnWorld.elapsedMs || rrWorld.elapsedMs, dqnWorld, rrWorld);
  showWinner(dqnWorld, rrWorld);

  // Trigger next decisions
  if (dqnWorld.awaitingNext && dqnWorld.autoNextPending && !dqnWorld.waitingServer && !dqnWorld.finished) {
    const tok = runToken;
    dqnRequestNext(tok);
  }
  if (rrWorld.awaitingNext && rrWorld.autoNextPending && !rrWorld.waitingServer && !rrWorld.finished) {
    rrTriggerNext();
  }

  const anyRunning = !dqnWorld.finished || !rrWorld.finished ||
    dqnWorld.movingCars.length > 0 || rrWorld.movingCars.length > 0;
  if (anyRunning) {
    animId = requestAnimationFrame(loop);
  }
}

// ─────────────────────────────────────────────────────────────
// Start both simulations
// ─────────────────────────────────────────────────────────────
function startBoth(initialCounts, dqnFirstPayload) {
  // --- Shared initial counts (same traffic for both) ---
  const counts = normalizeCounts(initialCounts);

  // Setup RR world with same counts
  Object.assign(rrWorld, createWorld());
  rrWorld.counts = { ...counts };
  rrWorld.initialCounts = { ...counts };
  rrWorld.totalInitial = totalCars(counts);
  rrWorld.seededRand = mulberry32(hashCounts(counts) ^ 0xdeadbeef);
  rrWorld.runMode = "rr";
  rrWorld.rrCursor = 0;

  // Setup DQN world
  applyDQNPayload(dqnFirstPayload, true);

  // Kick off RR first move
  applyRRDecision(rrWorld);

  // Init chart
  initChart();

  // Start animation
  if (animId) cancelAnimationFrame(animId);
  lastTs = 0;
  animId = requestAnimationFrame(loop);
}

// ─────────────────────────────────────────────────────────────
// Reset
// ─────────────────────────────────────────────────────────────
function resetAll() {
  runToken = 0;
  if (animId) cancelAnimationFrame(animId);
  animId = 0;
  lastTs = 0;

  Object.assign(dqnWorld, createWorld());
  Object.assign(rrWorld, createWorld());

  if (dqnCtx && dqnCanvas) {
    dqnCtx.clearRect(0, 0, dqnCanvas.width, dqnCanvas.height);
    drawScene(dqnCanvas, dqnCtx, dqnWorld, "#ff8d3a");
  }
  if (rrCtx && rrCanvas) {
    rrCtx.clearRect(0, 0, rrCanvas.width, rrCanvas.height);
    drawScene(rrCanvas, rrCtx, rrWorld, "#4dc0ff");
  }

  updateSimUI("dqn", dqnWorld, byId("dqnStatus"));
  updateSimUI("rr", rrWorld, byId("rrStatus"));
  updateScoreboard(dqnWorld, rrWorld);

  const dqnBadge = byId("dqnWinnerBadge");
  const rrBadge = byId("rrWinnerBadge");
  if (dqnBadge) { dqnBadge.classList.add("hidden"); dqnBadge.classList.remove("shown"); }
  if (rrBadge) { rrBadge.classList.add("hidden"); rrBadge.classList.remove("shown"); }

  setText(byId("dqnStatus"), "Waiting...");
  setText(byId("rrStatus"), "Waiting...");
  byId("dqnStatus")?.classList.remove("pill-done");
  byId("rrStatus")?.classList.remove("pill-done");

  if (chartInstance) { chartInstance.destroy(); chartInstance = null; }
}

// ─────────────────────────────────────────────────────────────
// Backend status probe
// ─────────────────────────────────────────────────────────────
async function probeBackend() {
  const chip = byId("backendStatusChip");
  const val = byId("backendStatusValue");
  const modeChip = byId("controlModeChip");
  const modeVal = byId("controlModeValue");

  const swapClass = (node, cls) => {
    node?.classList.remove("is-ok", "is-warn", "is-error", "is-mock");
    node?.classList.add(cls);
  };

  setText(val, "Checking...");
  swapClass(chip, "is-warn");

  try {
    const st = await fetchJson("/api/status", { method: "GET" });
    setText(val, "Online");
    swapClass(chip, "is-ok");
    setText(modeVal, String(st?.mode || "Unknown").toUpperCase());
    swapClass(modeChip, "is-ok");
  } catch (_) {
    setText(val, "Offline");
    swapClass(chip, "is-error");
    setText(modeVal, "Unavailable");
    swapClass(modeChip, "is-warn");
  }
}

// ─────────────────────────────────────────────────────────────
// File name display
// ─────────────────────────────────────────────────────────────
function bindFileListeners(form) {
  form.querySelectorAll("input[type='file'][name]").forEach((input) => {
    input.addEventListener("change", () => {
      const node = byId(`fn-${input.name}`);
      if (node) node.textContent = (input.files?.[0]?.name) || "No file selected";
    });
  });
}

// ─────────────────────────────────────────────────────────────
// Form submission
// ─────────────────────────────────────────────────────────────
const form = byId("cmpForm");
const startBtn = byId("cmpStartBtn");
const resetBtn = byId("cmpResetBtn");
const msgEl = byId("cmpMessage");
const stateChip = byId("cmpStateChip");
const stateVal = byId("cmpStateValue");

function setCmpState(text, cls = "is-warn") {
  setText(stateVal, text);
  stateChip?.classList.remove("is-ok", "is-warn", "is-error", "is-mock");
  stateChip?.classList.add(cls);
}
function setMsg(text, err = false) {
  if (!msgEl) return;
  msgEl.textContent = text || "";
  msgEl.style.color = err ? "#ef8f8f" : "#8cd1ff";
}

// Mock initial payload from form files (for offline fallback)
function buildMockPayload(formData) {
  let seed = 0x9e3779b9;
  LANE_KEYS.forEach((n, i) => {
    const f = formData.get(n);
    const s = f?.name || `${n}-none`;
    for (let j = 0; j < s.length; j++) seed ^= s.charCodeAt(j) + i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed ^= (f?.size || 0) + i * 17;
  });
  seed = seed >>> 0;
  const rand = mulberry32(seed || Date.now());
  const counts = {};
  LANE_KEYS.forEach((l) => {
    const f = formData.get(l);
    counts[l] = (f?.name?.length > 0) ? 4 + Math.floor(rand() * 20) : 0;
  });
  const lane = maxLane(counts);
  const maxQ = Math.max(...LANE_KEYS.map((l) => counts[l] || 0));
  const dur = clamp(Math.round(8 + (maxQ > 0 ? (counts[lane] / maxQ) : 0) * 22), 5, 45);
  const dir = lane.replace("lane", "");
  return {
    simulation: {
      initial_counts: { ...counts },
      selected_lane: lane,
      selected_direction: dir,
      selected_duration: dur,
      clear_rate_min: 1,
      clear_rate_max: 2,
      yellow_ms: 900,
      seed: seed,
      mode: "mock-fallback",
    },
    decision: { direction: dir, duration: dur, mode: "mock-fallback" },
  };
}

if (form) {
  bindFileListeners(form);

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (startBtn) startBtn.disabled = true;
    resetAll();

    runToken = Date.now();
    const localToken = runToken;
    const formData = new FormData(form);

    setCmpState("Running...", "is-warn");
    setMsg("Sending to backend and starting comparison...");

    let dqnPayload;
    try {
      dqnPayload = await callRunCycle(formData);
      if (localToken !== runToken) return;
      dqnWorld.runMode = "live";
      setCmpState("Live", "is-ok");
      setMsg("Live DQN running. Round-Robin running in parallel.");
    } catch (err) {
      if (localToken !== runToken) return;
      dqnPayload = buildMockPayload(formData);
      dqnWorld.runMode = "mock";
      setCmpState("Simulated", "is-mock");
      setMsg(`Backend offline: ${err.message}. Both sims in simulated mode.`, true);
    }

    startBoth(dqnPayload.simulation?.initial_counts || {}, dqnPayload);
    if (startBtn) startBtn.disabled = false;
  });
}

if (resetBtn) {
  resetBtn.addEventListener("click", () => {
    form?.reset();
    form?.querySelectorAll("input[type='file'][name]").forEach((inp) => {
      const node = byId(`fn-${inp.name}`);
      if (node) node.textContent = "No file selected";
    });
    resetAll();
    setMsg("Reset.");
    setCmpState("Idle", "is-warn");
  });
}

// Initial draw (empty scenes)
drawScene(dqnCanvas, dqnCtx, dqnWorld, "#ff8d3a");
drawScene(rrCanvas, rrCtx, rrWorld, "#4dc0ff");
probeBackend();
