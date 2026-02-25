# Roadmap

## Project Goal (End-of-Month MVP)

A demo that shows, on **real video or live camera** (even 1 camera is fine):

1) **Vehicle detection + per-lane counts**  
2) **Adaptive signal timing** (green time changes based on density)  
3) **Emergency detection trigger** (ambulance OR siren OR manual override)  
4) **Green corridor mode** (preemption sequence across 2–3 “intersections” in software + Arduino LEDs)  
5) **Logging + basic dashboard** (simple UI or web page)

You can still keep LSTM + DQN in the roadmap, but they are **not required for a solid month MVP**.

---

## Week 1 (Days 1–7): Repo setup + dataset + baseline detection

### Deliverables

- GitHub repo created: `adaptive-traffic-signal-green-corridor`
- Working YOLOv8 inference script on sample traffic video
- Dataset prepared in YOLO format (or at least verified)

### Tasks

1. **Repo structure (fixed early)**
   - `detection/` (YOLO inference + tracking/counting)
   - `control/` (signal state machine + corridor logic)
   - `hardware/` (Arduino + wiring docs)
   - `gui/` (optional week 3)
   - `data/` (placeholders + scripts; don’t commit huge datasets)
   - `docs/` (architecture + wiring + demo steps)

2. **Run YOLOv8 pretrained first**
   - Use `yolov8n.pt` or `yolov8s.pt` for speed.
   - Prove you can detect vehicles on a recorded intersection video.

3. **Define lanes/ROIs**
   - Hardcode 4 lane polygons or rectangular ROIs for counts.
   - Output: `counts = {N: x, S: y, E: z, W: w}` every second.

4. **Logging (start early)**
   - SQLite table: timestamp, counts, chosen_phase, emergency_flag.

**End of week demo:** A video window with bounding boxes + live counts per lane, writing to SQLite.

---

## Week 2 (Days 8–14): Adaptive signal control (no RL yet) + Arduino LEDs

### Deliverables

- Signal controller working in software (state machine)
- Arduino-controlled LEDs show correct phase transitions
- Density-based green time adaptation

### Tasks

1. **Signal state machine**
   - Phases: `NS_GREEN`, `NS_YELLOW`, `EW_GREEN`, `EW_YELLOW`
   - Min/max green bounds (e.g., 10s–60s)
   - Rule-based adaptive timing (fast to implement):
     - `green_time = clamp(k * lane_density_ratio, min_green, max_green)`
     - or choose next green based on max queue lane group

2. **Arduino interface**
   - Serial protocol: send commands like `NS_G`, `NS_Y`, `EW_G`, etc.
   - Arduino drives 12 LEDs (R/Y/G for each direction) or simplified 2-direction demo.

3. **Sensor placeholders**
   - IR/ultrasonic can be added later; for now, camera counts drive logic.

**End of week demo:** Vehicles counted → green time changes → LEDs match in real hardware.

---

## Week 3 (Days 15–21): Emergency priority + Green corridor mode (core novelty)

### Deliverables

- Emergency detection trigger works (choose at least 1 reliable method)
- Green corridor sequence demonstrated across multiple intersections (simulated or hardware)

### Choose emergency trigger (pick 1 now, add others later)

**Option A (fastest & reliable): manual override**

- Press a GUI button / keyboard key → corridor mode.

**Option B (vision): ambulance detection**

- If dataset lacks ambulances, use pretrained + quick fine-tune with a small custom ambulance set.

**Option C (audio): siren detection**

- KY-037 is noisy; best to treat audio as *support signal*, not sole trigger.

### Corridor logic (must be explicit)

- Maintain a list of intersections along a route: `I1 -> I2 -> I3`
- When emergency active:
  - Force “green wave” along route
  - Hold cross-traffic red
  - After timeout or vehicle passed, revert to adaptive mode

**End of week demo:** Trigger emergency → system preempts → corridor pattern appears (software + LEDs).

---

## Week 4 (Days 22–30): GUI + polish + performance + documentation + final demo

### Deliverables

- Simple dashboard (Tkinter or Flask)
- Demo script + recorded results
- Documentation good enough for submission/patent draft
- Optional: add LSTM predictor (if time)

### Tasks

1. **GUI (keep minimal)**
   - Live counts
   - Current signal phase + time remaining
   - Emergency status + “Trigger/Reset”
   - Config sliders: min/max green, YOLO conf

2. **Metrics**
   - Average wait proxy (queue length over time)
   - Emergency clearance time (trigger → corridor green)

3. **Packaging**
   - `requirements.txt`
   - `README.md` with “How to run demo”
   - Wiring diagram photos/notes

4. **Optional if time remains**
   - LSTM predictor: use logged counts to predict next 30–60s density
   - Use predictor to adjust green times proactively (even without RL)

**End-of-month demo:** One-click run + dashboard + LEDs + emergency corridor + logs.

---

## What to DEFER (so you finish in 1 month)

To hit the deadline, postpone these unless you’re ahead:

- Full DQN training (very time-consuming + needs simulation)
- Multi-camera (start with 1 camera, then expand)
- Full sensor fusion with ultrasonic/IR validation (nice-to-have)
- Model quantization/TFLite optimization (do later unless Pi is required)

---

## 3 Clarifying Questions (so I can tailor the plan precisely)

1) Are you required to run the final demo on **Raspberry Pi**, or is **laptop + Arduino** acceptable for this month?  
2) How many intersections do you need to demonstrate for “green corridor”: **2** is enough for proof—do you need more?  
3) Do you already have **any ambulance-labeled images**, or should we rely on **manual trigger + siren** for MVP?

Answer these and I’ll convert the plan into a **day-by-day checklist** and a **repo folder structure + milestones** you can follow exactly.
