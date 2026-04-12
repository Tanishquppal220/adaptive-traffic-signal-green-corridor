/**
 * compare.js — DQN vs Round-Robin side-by-side comparison
 *
 * Two fully independent simulation worlds share one rAF loop.
 * DQN calls the Flask backend (/api/run_cycle → /api/next_cycle).
 * Round-Robin is pure clientside: fixed 15s green in N→S→E→W order.
 * Queue Depth Over Time is recorded every ~1 s and shown as a line chart.
 */
"use strict";

// ── Constants ───────────────────────────────────────────────────────
const LANES      = ["laneN","laneS","laneE","laneW"];
const LABELS     = { laneN:"North", laneS:"South", laneE:"East", laneW:"West" };
const DIR_TO_LANE= { N:"laneN", S:"laneS", E:"laneE", W:"laneW" };
let   RR_GREEN_S = 15;    // mutable — controlled by the slider
const YELLOW_MS  = 900;
const CHART_SAMPLE_INTERVAL_MS = 1000;

// ── Small helpers ───────────────────────────────────────────────────
const $   = id => document.getElementById(id);
const qs  = sel => document.querySelector(sel);

function tx(id, v, fb = "—") {
  const el = $(id);
  if (el) el.textContent = (v === null || v === undefined || v === "") ? fb : String(v);
}

function clamp(v, lo, hi) { return Math.min(hi, Math.max(lo, v)); }

function normalize(raw = {}) {
  const o = {};
  LANES.forEach(l => { const n = Number(raw[l]); o[l] = Number.isFinite(n) ? Math.max(0, Math.floor(n)) : 0; });
  return o;
}

function totalQ(counts) { return LANES.reduce((s,l) => s + (counts[l]||0), 0); }

function mulberry32(seed) {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let x = Math.imul(t ^ (t>>>15), 1|t);
    x ^= x + Math.imul(x ^ (x>>>7), 61|x);
    return ((x ^ (x>>>14))>>>0) / 4294967296;
  };
}
function rInt(rng, lo, hi) { return Math.floor(rng()*(hi-lo+1))+lo; }

function laneFromDir(d) { return DIR_TO_LANE[String(d||"N").toUpperCase().trim()] || "laneN"; }

// ── World factory ───────────────────────────────────────────────────
function mkWorld(prefix, canvasId) {
  const canvas = $(canvasId);
  return {
    prefix, canvas, ctx: canvas ? canvas.getContext("2d") : null,
    counts:     { laneN:0, laneS:0, laneE:0, laneW:0 },
    initCounts: { laneN:0, laneS:0, laneE:0, laneW:0 },
    movingCars: [],
    activeLane: null,
    signal:     "red",
    phaseType:  "idle",   // idle | fixed-green | yellow | paused | done
    phaseTimeMs:0,
    clearAccMs: 0,
    assignedMs: 0,
    yellowMs:   YELLOW_MS,
    clearRateMin:1, clearRateMax:2,
    phaseCycles:{ laneN:0, laneS:0, laneE:0, laneW:0 },
    elapsedMs:  0,
    finished:   false,
    finishMs:   null,
    rng:        Math.random,
    awaitingNext:false,
    autoNextPending:false,
    waitingServer:false,
    lastServedLane:null,
    runMode:"idle",
    // time-series
    timeHistory: [],         // [{t_s, q}]
    lastSampleMs: 0,
  };
}

const dqnW = mkWorld("dqn","dqnCanvas");
const rrW  = mkWorld("rr", "rrCanvas");

// ── Round-Robin state ───────────────────────────────────────────────
let rrSlot = 0;

// ── Canvas drawing ──────────────────────────────────────────────────
function geo(w) {
  if (!w.canvas) return null;
  const cw=w.canvas.width, ch=w.canvas.height;
  const cx=cw/2, cy=ch/2;
  const roadW=Math.min(cw,ch)*0.38, half=roadW/2;
  return { cx,cy,roadW,half,W:cw,H:ch,
    xN:cx-18, xS:cx+18, yE:cy-18, yW:cy+18,
    stopN:cy-half-8, stopS:cy+half+8, stopE:cx+half+8, stopW:cx-half-8 };
}

function qPos(lane,i,g) {
  const sp=28;
  if(lane==="laneN") return {x:g.xN, y:g.stopN-22-i*sp, a:Math.PI/2};
  if(lane==="laneS") return {x:g.xS, y:g.stopS+22+i*sp, a:-Math.PI/2};
  if(lane==="laneE") return {x:g.stopE+22+i*sp, y:g.yE, a:Math.PI};
  return {x:g.stopW-22-i*sp, y:g.yW, a:0};
}

function mPath(lane,g) {
  if(lane==="laneN") return {sx:g.xN,sy:g.stopN-6,ex:g.xN,ey:g.H+40,a:Math.PI/2};
  if(lane==="laneS") return {sx:g.xS,sy:g.stopS+6,ex:g.xS,ey:-40,a:-Math.PI/2};
  if(lane==="laneE") return {sx:g.stopE+6,sy:g.yE,ex:-40,ey:g.yE,a:Math.PI};
  return {sx:g.stopW-6,sy:g.yW,ex:g.W+40,ey:g.yW,a:0};
}

function spawnCar(w, lane) {
  const pal=["#2a9d8f","#2b8bd3","#915bb8","#c77d2d","#3c5fcb"];
  w.movingCars.push({ lane, t:0, speed:0.35+w.rng()*0.25, color:pal[Math.floor(w.rng()*pal.length)] });
}

function roundRect(ctx,x,y,bw,bh,r) {
  const rd=Math.min(r,bw/2,bh/2);
  ctx.beginPath();
  ctx.moveTo(x+rd,y); ctx.arcTo(x+bw,y,x+bw,y+bh,rd);
  ctx.arcTo(x+bw,y+bh,x,y+bh,rd); ctx.arcTo(x,y+bh,x,y,rd);
  ctx.arcTo(x,y,x+bw,y,rd); ctx.closePath();
}

function drawCar(ctx,x,y,a,col) {
  ctx.save(); ctx.translate(x,y); ctx.rotate(a);
  ctx.fillStyle=col; roundRect(ctx,-12,-7,24,14,4); ctx.fill();
  ctx.fillStyle="rgba(255,255,255,0.6)"; ctx.fillRect(-7,-5,14,3);
  ctx.restore();
}

function drawScene(w) {
  const ctx=w.ctx; if(!ctx||!w.canvas) return;
  const g=geo(w); if(!g) return;

  ctx.clearRect(0,0,g.W,g.H);
  ctx.fillStyle="#1c2530"; ctx.fillRect(0,0,g.W,g.H);

  // roads
  ctx.fillStyle="#333d47";
  ctx.fillRect(g.cx-g.half,0,g.roadW,g.H);
  ctx.fillRect(0,g.cy-g.half,g.W,g.roadW);

  // dashes
  ctx.strokeStyle="rgba(240,240,240,0.45)"; ctx.lineWidth=2; ctx.setLineDash([10,10]);
  ctx.beginPath();
  ctx.moveTo(g.cx,0); ctx.lineTo(g.cx,g.H);
  ctx.moveTo(0,g.cy); ctx.lineTo(g.W,g.cy);
  ctx.stroke(); ctx.setLineDash([]);

  // stop lines
  ctx.strokeStyle="#f9f3cc"; ctx.lineWidth=2.5;
  ctx.beginPath();
  ctx.moveTo(g.cx-g.half,g.stopN); ctx.lineTo(g.cx+g.half,g.stopN);
  ctx.moveTo(g.cx-g.half,g.stopS); ctx.lineTo(g.cx+g.half,g.stopS);
  ctx.moveTo(g.stopW,g.cy-g.half); ctx.lineTo(g.stopW,g.cy+g.half);
  ctx.moveTo(g.stopE,g.cy-g.half); ctx.lineTo(g.stopE,g.cy+g.half);
  ctx.stroke();

  // queue cars
  LANES.forEach(lane => {
    const cnt=w.counts[lane]||0, show=Math.min(cnt,14);
    for(let i=0;i<show;i++){ const p=qPos(lane,i,g); drawCar(ctx,p.x,p.y,p.a,"#4a90d9"); }
    if(cnt>show) {
      const p=qPos(lane,show,g);
      ctx.fillStyle="#fff"; ctx.font="11px IBM Plex Mono,monospace";
      ctx.fillText(`+${cnt-show}`,p.x+10,p.y+3);
    }
  });

  // moving cars
  w.movingCars.forEach(car => {
    const p=mPath(car.lane,g);
    drawCar(ctx,p.sx+(p.ex-p.sx)*car.t, p.sy+(p.ey-p.sy)*car.t, p.a, car.color);
  });

  // traffic lights
  const lp={
    laneN:{x:g.xN+30,y:g.stopN+16}, laneS:{x:g.xS-30,y:g.stopS-16},
    laneE:{x:g.stopE-16,y:g.yE-30}, laneW:{x:g.stopW+16,y:g.yW+30}
  };
  LANES.forEach(lane => {
    const p=lp[lane], act=w.activeLane===lane;
    ctx.fillStyle="#111"; ctx.fillRect(p.x-8,p.y-22,16,42);
    [{y:-14,c:"#d64545",on:!act||w.signal==="red"},
     {y:-2, c:"#d9a404",on:act&&w.signal==="yellow"},
     {y:10, c:"#2f9e44",on:act&&w.signal==="green"}]
    .forEach(l=>{
      ctx.beginPath(); ctx.arc(p.x,p.y+l.y,4.2,0,Math.PI*2);
      ctx.fillStyle=l.on?l.c:"#3d3d3d"; ctx.fill();
    });
  });

  // done overlay
  if(w.finished) {
    ctx.fillStyle="rgba(7,16,24,0.58)"; ctx.fillRect(0,0,g.W,g.H);
    ctx.textAlign="center";
    ctx.font="bold 26px Space Grotesk,sans-serif"; ctx.fillStyle="#2f9e44";
    ctx.fillText("✓ All Cleared!",g.cx,g.cy);
    ctx.font="14px IBM Plex Mono,monospace"; ctx.fillStyle="rgba(255,255,255,0.65)";
    ctx.fillText(`Time: ${(w.elapsedMs/1000).toFixed(1)}s`,g.cx,g.cy+30);
    ctx.textAlign="left";
  }
}

// ── Phase engine ────────────────────────────────────────────────────
function enterGreen(w,lane) {
  w.activeLane=lane; w.signal="green"; w.phaseType="fixed-green";
  w.phaseTimeMs=w.assignedMs; w.clearAccMs=0;
  w.phaseCycles[lane]=(w.phaseCycles[lane]||0)+1;
  w.lastServedLane=lane; w.awaitingNext=false;
}

function enterYellow(w) { w.signal="yellow"; w.phaseType="yellow"; w.phaseTimeMs=w.yellowMs; }

function finishSim(w) {
  w.signal="red"; w.activeLane=null; w.phaseType="done";
  w.finished=true; w.autoNextPending=false; w.finishMs=w.elapsedMs;
}

function pauseForNext(w) {
  w.signal="red"; w.activeLane=null; w.phaseType="paused"; w.awaitingNext=true;
  if(totalQ(w.counts)<=0) { finishSim(w); return; }
  w.autoNextPending=true;
}

function tickPhase(w,dt) {
  if(["done","idle","paused"].includes(w.phaseType)) return;

  if(w.signal==="green" && w.activeLane) {
    w.clearAccMs+=dt;
    while(w.clearAccMs>=1000) {
      w.clearAccMs-=1000;
      const l=w.activeLane;
      if((w.counts[l]||0)>0) {
        const rm=Math.min(w.counts[l],rInt(w.rng,w.clearRateMin,w.clearRateMax));
        w.counts[l]-=rm;
        for(let i=0;i<rm;i++) spawnCar(w,l);
      }
    }
  }

  if(w.phaseType==="fixed-green") {
    w.phaseTimeMs-=dt; if(w.phaseTimeMs<=0) enterYellow(w);
  } else if(w.phaseType==="yellow") {
    w.phaseTimeMs-=dt; if(w.phaseTimeMs<=0) pauseForNext(w);
  }
}

function tickMovingCars(w,dt) {
  w.movingCars.forEach(c=>{ c.t+=(c.speed*dt)/1000; });
  w.movingCars=w.movingCars.filter(c=>c.t<1.05);
}

// ── Sample time-series ──────────────────────────────────────────────
function sampleTimeSeries(w) {
  const t = w.elapsedMs / 1000;
  w.timeHistory.push({ t, q: totalQ(w.counts) });
}

// ── Mock DQN decision ───────────────────────────────────────────────
let mockCursor = 0;
function pickMockLane(counts) {
  const mx=Math.max(...LANES.map(l=>counts[l]||0));
  const cands=LANES.filter(l=>(counts[l]||0)===mx);
  if(!cands.length) return "laneN";
  for(let i=0;i<LANES.length;i++){
    const idx=(mockCursor+i)%LANES.length;
    if(cands.includes(LANES[idx])){ mockCursor=(idx+1)%LANES.length; return LANES[idx]; }
  }
  return cands[0];
}

function mockDur(q,mx) { if(mx<=0||q<=0) return 8; return clamp(Math.round(8+(q/mx)*22),5,45); }

function buildMockDecision(counts) {
  const lane=pickMockLane(counts);
  const mx=Math.max(...LANES.map(l=>counts[l]||0));
  return { lane, durationSec: mockDur(counts[lane]||0, mx) };
}

// ── Round-Robin ─────────────────────────────────────────────────────
function nextRRLane() {
  const lane=LANES[rrSlot % LANES.length];
  rrSlot=(rrSlot+1)%LANES.length;
  return lane;
}

function applyRRStep(w) {
  const lane=nextRRLane();
  w.assignedMs=RR_GREEN_S*1000;
  w.awaitingNext=false; w.autoNextPending=false;
  enterGreen(w,lane);
  tx("rr-sched-lane", LABELS[lane]||lane);
  setStatusPill("rr",`${LABELS[lane]} — ${RR_GREEN_S}s`);
}

// ── Backend fetch ───────────────────────────────────────────────────
async function fetchJ(url,opts={}) {
  const r=await fetch(url,opts);
  let p=null;
  try { p=await r.json(); } catch(_) {}
  if(!r.ok) throw new Error(p?.error||`${r.status} ${r.statusText}`);
  return p||{};
}

function parseDqn(payload) {
  const sim=payload?.simulation||{}, dec=payload?.decision||{};
  const lane=sim.selected_lane||laneFromDir(dec.direction);
  const dur=clamp(Math.round(Number(sim.selected_duration||dec.duration||10)),5,60);
  const init=sim.initial_counts?normalize(sim.initial_counts):null;
  return { lane, durationSec:dur, init, seed:sim.seed, mode:dec.mode||"dqn" };
}

// ── Apply decision to world ─────────────────────────────────────────
function applyDecision(w,lane,durSec,seed=null) {
  if(seed!=null) w.rng=mulberry32(seed);
  w.assignedMs=durSec*1000;
  w.awaitingNext=false; w.autoNextPending=false;
  enterGreen(w,lane);
}

function initWorld(w,counts,seed=null) {
  w.counts     = normalize(counts);
  w.initCounts = {...w.counts};
  w.movingCars = [];
  w.phaseCycles= { laneN:0,laneS:0,laneE:0,laneW:0 };
  w.finished=false; w.finishMs=null; w.elapsedMs=0;
  w.signal="red"; w.phaseType="idle"; w.activeLane=null;
  w.awaitingNext=false; w.autoNextPending=false; w.waitingServer=false;
  w.lastServedLane=null; w.assignedMs=0;
  w.timeHistory=[]; w.lastSampleMs=0;
  if(seed!=null) w.rng=mulberry32(seed);
}

// ── Mock counts from upload ─────────────────────────────────────────
function hashFiles(fd) {
  let seed=0x9e3779b9;
  [...LANES,"sirenAudio"].forEach((name,idx)=>{
    const f=fd.get(name);
    const s=(f&&f.name)?f.name:`${name}-none`;
    const sz=(f&&Number.isFinite(f.size))?f.size:0;
    for(let i=0;i<s.length;i++) seed^=s.charCodeAt(i)+idx+0x9e3779b9+(seed<<6)+(seed>>2);
    seed^=sz+idx*17;
  });
  return seed>>>0;
}

function mockCountsFromUpload(fd) {
  const rng=mulberry32(hashFiles(fd)||Date.now());
  const c={};
  LANES.forEach(l=>{ const f=fd.get(l); c[l]=(f&&f.name?4+Math.floor(rng()*22):0); });
  return c;
}

// ── UI helpers ──────────────────────────────────────────────────────
function setMsg(text,err=false) {
  const el=$("cmpMessage"); if(!el) return;
  el.textContent=text; el.style.color=err?"#ef8f8f":"#8cd1ff";
}

function setStatusPill(prefix,text,type="running") {
  const el=$(prefix==="dqn"?"dqnStatusPill":"rrStatusPill"); if(!el) return;
  el.textContent=text;
  el.className="c-sim-status-pill"+(type?" "+type:"");
}

function chipState(id,cls) {
  const el=$(id); if(!el) return;
  el.className="c-status-chip "+cls;
}

function setComparisonState(text,cls="warn") {
  tx("val-state",text); chipState("chip-state",cls);
}

function renderPhase(w) {
  const p=w.prefix;
  tx(`${p}-ph-lane`, w.activeLane?LABELS[w.activeLane]:"-", "-");
  tx(`${p}-ph-signal`, String(w.signal||"RED").toUpperCase());

  const active=["fixed-green","yellow"].includes(w.phaseType);
  const remSec = active ? Math.max(0, Math.ceil(w.phaseTimeMs/1000)) : null;

  // Signal phase remaining
  tx(`${p}-ph-rem`, remSec!==null ? `${remSec}s` : "-");

  // DQN: live countdown in Duration field
  if(p==="dqn") {
    const durEl=$("dqn-dec-dur");
    if(durEl) {
      if(active && remSec!==null) {
        durEl.textContent=`${remSec}s`;
        durEl.classList.add("counting");
      } else {
        durEl.classList.remove("counting");
      }
    }
  }

  // RR: live countdown in Remaining field
  if(p==="rr") {
    tx("rr-sched-dur", remSec!==null ? `${remSec}s` : "-");
  }
}

function renderQueue(w) {
  LANES.forEach(l => tx(`${w.prefix}-q-${l}`, Math.max(0,w.counts[l]||0), "0"));
}

// ── Scoreboard ──────────────────────────────────────────────────────
function carsCleared(w) {
  return Math.max(0, totalQ(w.initCounts) - totalQ(w.counts));
}

function throughput(w) {
  const cl=carsCleared(w);
  if(cl===0||w.elapsedMs<500) return null;
  return (cl/(w.elapsedMs/1000)).toFixed(2);
}

function renderScoreboard() {
  const totalInit=totalQ(dqnW.initCounts)||totalQ(rrW.initCounts)||1;

  tx("sb-dqn-elapsed",`${(dqnW.elapsedMs/1000).toFixed(1)}s`);
  tx("sb-rr-elapsed", `${(rrW.elapsedMs/1000).toFixed(1)}s`);

  const dCl=carsCleared(dqnW), rCl=carsCleared(rrW);
  tx("sb-dqn-cleared", dCl);
  tx("sb-rr-cleared",  rCl);

  tx("sb-dqn-cycles", LANES.reduce((s,l)=>s+(dqnW.phaseCycles[l]||0),0));
  tx("sb-rr-cycles",  LANES.reduce((s,l)=>s+(rrW.phaseCycles[l]||0),0));

  tx("sb-dqn-queue", totalQ(dqnW.counts));
  tx("sb-rr-queue",  totalQ(rrW.counts));

  const dTp=throughput(dqnW), rTp=throughput(rrW);
  tx("sb-dqn-throughput", dTp?`${dTp}/s`:"—");
  tx("sb-rr-throughput",  rTp?`${rTp}/s`:"—");

  // Progress bar — proportional to cars cleared
  const dPct=Math.min(50, (dCl/totalInit)*50);
  const rPct=Math.min(50, (rCl/totalInit)*50);
  const pbD=$("pbDqn"), pbR=$("pbRr");
  if(pbD) pbD.style.width=`${dPct}%`;
  if(pbR) pbR.style.width=`${rPct}%`;

  // Winner badges
  if(dqnW.finished && rrW.finished) {
    if(dqnW.finishMs <= rrW.finishMs) {
      $("dqnWinnerTag")?.classList.remove("hidden");
    } else {
      $("rrWinnerTag")?.classList.remove("hidden");
    }
  } else if(dqnW.finished) {
    $("dqnWinnerTag")?.classList.remove("hidden");
  } else if(rrW.finished) {
    $("rrWinnerTag")?.classList.remove("hidden");
  }
}

// ── Panel glow on finish ────────────────────────────────────────────
function applyFinishVisuals() {
  if(!dqnW.finished && !rrW.finished) return;

  if(dqnW.finished && rrW.finished) {
    const dWins = dqnW.finishMs <= rrW.finishMs;
    qs(".dqn-panel")?.classList.toggle("sim-won",  dWins);
    qs(".dqn-panel")?.classList.toggle("sim-done", !dWins);
    qs(".rr-panel")?.classList.toggle("sim-won",  !dWins);
    qs(".rr-panel")?.classList.toggle("sim-done",  dWins);
    setComparisonState(dWins?"DQN Won 🏆":"Round-Robin Won 🏆", dWins?"ok":"mock");
  } else if(dqnW.finished) {
    qs(".dqn-panel")?.classList.add("sim-won");
    setComparisonState("DQN Finished First","ok");
  } else if(rrW.finished) {
    qs(".rr-panel")?.classList.add("sim-won");
    setComparisonState("RR Finished First","mock");
  }
}

// ── Time chart (line) ───────────────────────────────────────────────
let timeChart = null;

function initTimeChart() {
  const canvas=$("timeChart"); if(!canvas||typeof Chart==="undefined") return;
  if(timeChart){ timeChart.destroy(); timeChart=null; }

  timeChart=new Chart(canvas,{
    type:"line",
    data:{
      datasets:[
        {
          label:"DQN",
          data:[],
          borderColor:"#ff8d3a",
          backgroundColor:"rgba(255,141,58,0.08)",
          borderWidth:2.5,
          pointRadius:0,
          tension:0.35,
          fill:true,
        },
        {
          label:"Round-Robin",
          data:[],
          borderColor:"#4dc0ff",
          backgroundColor:"rgba(77,192,255,0.07)",
          borderWidth:2.5,
          pointRadius:0,
          tension:0.35,
          fill:true,
        },
      ],
    },
    options:{
      responsive:true,
      maintainAspectRatio:false,
      animation:{duration:0},
      interaction:{ mode:"index", intersect:false },
      plugins:{
        legend:{
          labels:{ color:"#9ab0c5", font:{family:"Space Grotesk"}, usePointStyle:true, pointStyleWidth:10 }
        },
        tooltip:{
          callbacks:{
            title: items => `t = ${Number(items[0].parsed.x).toFixed(1)}s`,
            label: ctx  => ` ${ctx.dataset.label}: ${ctx.parsed.y} cars`,
          },
          backgroundColor:"rgba(10,20,30,0.92)",
          titleColor:"#cde",
          bodyColor:"#aac",
          borderColor:"rgba(100,140,170,0.3)",
          borderWidth:1,
        },
      },
      scales:{
        x:{
          type:"linear",
          title:{ display:true, text:"Time (s)", color:"#7a9ab5", font:{size:11} },
          ticks:{ color:"#7a9ab5", stepSize:5 },
          grid:{ color:"rgba(100,140,170,0.1)" },
        },
        y:{
          beginAtZero:true,
          title:{ display:true, text:"Total Cars Queued", color:"#7a9ab5", font:{size:11} },
          ticks:{ color:"#7a9ab5" },
          grid:{ color:"rgba(100,140,170,0.1)" },
        },
      },
    },
  });
}

function pushTimeChart() {
  if(!timeChart) return;
  timeChart.data.datasets[0].data = dqnW.timeHistory.map(d=>({x:d.t,y:d.q}));
  timeChart.data.datasets[1].data = rrW.timeHistory.map(d=>({x:d.t,y:d.q}));
  timeChart.update("none");
}

// ── Animation loop ──────────────────────────────────────────────────
let animId=0, lastTs=0, running=false;

function animLoop(ts) {
  if(!running) return;
  if(!lastTs) lastTs=ts;
  const dt=Math.min(50,ts-lastTs); lastTs=ts;

  [dqnW,rrW].forEach(w=>{
    if(!w.finished) {
      w.elapsedMs+=dt;

      // sample time-series every second
      if(w.elapsedMs-w.lastSampleMs >= CHART_SAMPLE_INTERVAL_MS) {
        w.lastSampleMs=w.elapsedMs;
        sampleTimeSeries(w);
      }
    }

    tickPhase(w,dt);
    tickMovingCars(w,dt);
    drawScene(w);
    renderPhase(w);
    renderQueue(w);

    // auto-next decisions
    if(w.awaitingNext && !w.finished && w.autoNextPending) {
      if(totalQ(w.counts)<=0) { w.autoNextPending=false; finishSim(w); return; }
      w.autoNextPending=false;
      if(w.prefix==="dqn") scheduleDqnNext();
      else applyRRStep(w);
    }
  });

  renderScoreboard();
  applyFinishVisuals();
  pushTimeChart();

  const allDone=dqnW.finished&&rrW.finished;
  const moving=dqnW.movingCars.length>0||rrW.movingCars.length>0;
  if(!allDone||moving) {
    animId=requestAnimationFrame(animLoop);
  } else {
    running=false;
    setMsg("✓ Both simulations complete. See scoreboard and chart above.");
    setStatusPill("dqn","Finished","done");
    setStatusPill("rr", "Finished","done");
  }
}

function startLoop() {
  if(animId) cancelAnimationFrame(animId);
  lastTs=0; running=true;
  animId=requestAnimationFrame(animLoop);
}

// ── DQN next-cycle ──────────────────────────────────────────────────
async function scheduleDqnNext() {
  if(dqnW.waitingServer||dqnW.finished) return;

  if(dqnW.runMode==="mock") {
    const {lane,durationSec}=buildMockDecision(dqnW.counts);
    applyDecision(dqnW,lane,durationSec);
    updateDqnDecisionUI(lane,durationSec,"mock-fallback");
    setStatusPill("dqn",`Mock → ${LABELS[lane]} ${durationSec}s`);
    return;
  }

  dqnW.waitingServer=true;
  setStatusPill("dqn","⏳ Deciding…");
  try {
    const payload=await fetchJ("/api/next_cycle",{
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({lane_counts:{...dqnW.counts},current_active_lane:dqnW.lastServedLane}),
    });
    const {lane,durationSec,mode}=parseDqn(payload);
    applyDecision(dqnW,lane,durationSec);
    updateDqnDecisionUI(lane,durationSec,mode||"dqn");
    setStatusPill("dqn",`DQN → ${LABELS[lane]} ${durationSec}s`);
  } catch(err) {
    setMsg(`DQN fallback: ${err.message}`,true);
    dqnW.runMode="mock";
    chipState("chip-backend","warn");
    const {lane,durationSec}=buildMockDecision(dqnW.counts);
    applyDecision(dqnW,lane,durationSec);
    updateDqnDecisionUI(lane,durationSec,"mock");
  } finally {
    dqnW.waitingServer=false;
  }
}

function updateDqnDecisionUI(lane,dur) {
  tx("dqn-dec-dir",  LABELS[lane]||lane);
  // Duration is now driven live by renderPhase countdown — just set initial value
  const durEl=$("dqn-dec-dur");
  if(durEl && !durEl.classList.contains("counting")) durEl.textContent=`${dur}s`;
}

// ── Reset ───────────────────────────────────────────────────────────
function resetAll() {
  running=false;
  if(animId) cancelAnimationFrame(animId);
  animId=0; lastTs=0;
  rrSlot=0; mockCursor=0;

  [dqnW,rrW].forEach(w=>{
    w.counts={laneN:0,laneS:0,laneE:0,laneW:0};
    w.initCounts={...w.counts};
    w.movingCars=[]; w.activeLane=null; w.signal="red"; w.phaseType="idle";
    w.phaseTimeMs=0; w.clearAccMs=0; w.phaseCycles={laneN:0,laneS:0,laneE:0,laneW:0};
    w.elapsedMs=0; w.finished=false; w.finishMs=null;
    w.awaitingNext=false; w.autoNextPending=false; w.waitingServer=false;
    w.lastServedLane=null; w.assignedMs=0; w.runMode="idle";
    w.timeHistory=[]; w.lastSampleMs=0;
    renderPhase(w); renderQueue(w); drawScene(w);
    setStatusPill(w.prefix,"Waiting…","");
  });

  qs(".dqn-panel")?.classList.remove("sim-won","sim-done");
  qs(".rr-panel")?.classList.remove("sim-won","sim-done");
  $("dqnWinnerTag")?.classList.add("hidden");
  $("rrWinnerTag")?.classList.add("hidden");

  const pbD=$("pbDqn"),pbR=$("pbRr");
  if(pbD) pbD.style.width="50%";
  if(pbR) pbR.style.width="50%";

  setComparisonState("Idle","warn");
  setMsg("Form reset.");
  initTimeChart();
}

// ── Backend probe ───────────────────────────────────────────────────
async function probeBackend() {
  tx("val-backend","Checking…"); chipState("chip-backend","warn");
  tx("val-mode","—");
  try {
    const s=await fetchJ("/api/status");
    tx("val-backend","Online"); chipState("chip-backend","ok");
    tx("val-mode",(s?.mode||"unknown").toUpperCase());
  } catch(_) {
    tx("val-backend","Offline"); chipState("chip-backend","err");
    tx("val-mode","Unavailable");
  }
}

// ── Form submit ─────────────────────────────────────────────────────
const form=$("cmpForm");
const startBtn=$("cmpStartBtn");
const resetBtn=$("cmpResetBtn");

if(form) {
  form.addEventListener("submit", async evt => {
    evt.preventDefault();
    if(startBtn) startBtn.disabled=true;
    setMsg("Running DQN pipeline…");
    setComparisonState("Running…","ok");

    const fd=new FormData(form);
    let initCounts, dqnLane, dqnDur, dqnMode="dqn", dqnSeed;

    // ── Try live DQN ──────────────────────────────────────
    try {
      const payload=await fetchJ("/api/run_cycle",{method:"POST",body:fd});
      const parsed=parseDqn(payload);
      dqnLane=parsed.lane; dqnDur=parsed.durationSec;
      dqnMode=parsed.mode||"dqn"; dqnSeed=parsed.seed;
      initCounts=parsed.init||normalize(mockCountsFromUpload(fd));
      dqnW.runMode="live";
      chipState("chip-backend","ok");
      setMsg("DQN pipeline OK. Both simulations starting…");
    } catch(err) {
      dqnW.runMode="mock";
      chipState("chip-backend","err");
      initCounts=normalize(mockCountsFromUpload(fd));
      const md=buildMockDecision(initCounts);
      dqnLane=md.lane; dqnDur=md.durationSec; dqnMode="mock-fallback";
      dqnSeed=Date.now();
      setMsg(`Backend unavailable (${err.message}). Running mock fallback.`,true);
    }

    const seed=dqnSeed??Date.now();

    // ── Init DQN world ────────────────────────────────────
    initWorld(dqnW, initCounts, seed);
    applyDecision(dqnW, dqnLane, dqnDur, seed);
    updateDqnDecisionUI(dqnLane, dqnDur, dqnMode);
    setStatusPill("dqn",`DQN → ${LABELS[dqnLane]} ${dqnDur}s`);

    // ── Init RR world ─────────────────────────────────────
    rrSlot=0;
    initWorld(rrW, initCounts, seed+1);
    rrW.runMode="rr";
    applyRRStep(rrW);

    // reset panel classes & badges
    qs(".dqn-panel")?.classList.remove("sim-won","sim-done");
    qs(".rr-panel")?.classList.remove("sim-won","sim-done");
    $("dqnWinnerTag")?.classList.add("hidden");
    $("rrWinnerTag")?.classList.add("hidden");

    // ── Chart ─────────────────────────────────────────────
    initTimeChart();

    // ── Start loop ────────────────────────────────────────
    startLoop();
    if(startBtn) startBtn.disabled=false;
  });
}

if(resetBtn) {
  resetBtn.addEventListener("click",()=>{
    if(form) form.reset();
    [...LANES,"sirenAudio"].forEach(name=>{
      const el=$(`fn-${name}`); if(el) el.textContent="No file selected";
    });
    resetAll();
  });
}

// File name labels
if(form) {
  form.querySelectorAll("input[type='file'][name]").forEach(inp=>{
    inp.addEventListener("change",()=>{
      const el=$(`fn-${inp.name}`); if(!el) return;
      el.textContent=(inp.files&&inp.files[0])?inp.files[0].name:"No file selected";
    });
  });
}

// ── RR Duration Slider ──────────────────────────────────────────────
const rrSlider=$("rrDurSlider");
const rrDurDisplay=$("rrDurDisplay");
if(rrSlider) {
  function updateSliderUI() {
    const v=Number(rrSlider.value);
    RR_GREEN_S=v;
    if(rrDurDisplay) rrDurDisplay.textContent=`${v}s`;
    // update slider fill gradient
    const pct=((v-5)/(60-5)*100).toFixed(1);
    rrSlider.style.setProperty("--pct",`${pct}%`);
  }
  rrSlider.addEventListener("input", updateSliderUI);
  updateSliderUI();
}

// ── Boot ────────────────────────────────────────────────────────────
resetAll();
probeBackend();
