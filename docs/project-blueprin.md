# project-blueprint

# ADAPTIVE TRAFFIC SIGNAL & EMERGENCY CORRIDOR MANAGEMENT SYSTEM

## Project Blueprint: Bird’s Eye View

---

## 1. SYSTEM ARCHITECTURE BLUEPRINT

### 1.1 Internal Flow Diagram (Sequential)

```markdown
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SENSOR INPUT LAYER                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [Camera 1]  [Camera 2]  [Camera 3]  [Camera 4]  (4-way intersection)       │
│       ↓            ↓            ↓            ↓                              │
│  [IR Sensors] [Ultrasonic] [Sound Sensor] [RTC Module] [LDR Sensor]         │
│  (4 lanes)    (4 lanes)    (Siren detect) (Timestamp)  (Light level)        │
│                                                                             │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
                               ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│              PROCESSING LAYER 1: COMPUTER VISION & DETECTION                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────-─────┐    │
│  │ YOLOv8 Vehicle Detection Model (Main)                               │    │
│  │ • Input: 4 camera streams (640×480 @ 30fps)                         │    │
│  │ • Output: Bounding boxes + class labels (car/bike/bus/truck)        │    │
│  │ • Classes: 0=car, 1=bike, 2=bus, 3=truck, 4=auto                    │    │
│  │ • Processing: 15-30 FPS per camera (total ~60 FPS with 4 cameras)   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                 │                                           │
│                                 ↓                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Emergency Vehicle Classifier (YOLOv8 Fine-tuned)                    │    │
│  │ • Input: Cropped vehicle images from main YOLO                      │    │
│  │ • Output: Class confidence (ambulance/fire-truck/police/normal)     │    │
│  │ • Threshold: >0.8 confidence for emergency classification           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                 │                                           │
│                                 ↓                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Traffic Density Calculator                                          │    │
│  │ • Input: YOLO detections per lane + Ultrasonic queue length         │    │
│  │ • Calculation: vehicle_count / lane_area = density (vehicles/m²)    │    │
│  │ • Output: Density vector [North, South, East, West]                 │    │
│  │ • Validation: IR sensors confirm vehicle presence                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                 │                                           │
│                                 ↓                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Acoustic Siren Detection (Novel Component)                          │    │
│  │ • Input: Sound Sensor data (KY-037 audio frequency)                 │    │
│  │ • Detection: Frequency band 700-900 Hz (ambulance siren)            │    │
│  │ • Output: Boolean + confidence score for emergency siren            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
                               ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│              PROCESSING LAYER 2: MACHINE LEARNING & PREDICTION              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Traffic Density Predictor (Time-Series Model)                       │    │
│  │ Model Type: LSTM (Long Short-Term Memory) or XGBoost                │    │
│  │ • Input: Historical density data (past 5 minutes) + current density │    │
│  │         + time-of-day + day-of-week features                        │    │
│  │ • Training Data: 30-60 days of traffic patterns                     │    │
│  │ • Output: Predicted density for next 60 seconds (4 directions)      │    │
│  │ • Accuracy Target: MAE < 2 vehicles per lane                        │    │
│  │ • Update Frequency: Every 5 seconds                                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                 │                                           │
│                                 ↓                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Adaptive Signal Timing Optimizer (Reinforcement Learning)           │    │
│  │ Model Type: Q-Learning / Deep Q-Network (DQN)                       │    │
│  │ • Input: Current & predicted density, signal state, wait times      │    │
│  │ • States: Current signal configuration (RRYY, YYGG, etc.)           │    │
│  │ • Actions: Extend green, switch phase, adjust yellow duration       │    │
│  │ • Reward: Negative wait time + vehicle throughput                   │    │
│  │ • Output: Optimal green time per direction (5-60 seconds)           │    │
│  │ • Constraints: Yellow always 3-5s, cycle time 60-120s               │    │
│  │ • Update Frequency: Every 2-5 seconds                               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                 │                                           │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
                               ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│              DECISION LAYER: EMERGENCY VS NORMAL MODE                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  IF (Emergency Vehicle Detected)                                            │
│     ├─ Visual Detection: YOLO classifier confidence > 0.8                   │
│     ├─ Audio Detection: Siren frequency detected in sound sensor            │
│     ├─ Distance Check: Vehicle within 500m (GPS or speed estimation)        │
│     └─ Confirmation: 2+ signals must be true (multi-modal fusion)           │
│          │                                                                  │
│          ├→ [EMERGENCY MODE ACTIVATED]                                      │
│          │   └─ Override normal signal timing                               │
│          │   └─ Create synchronized green corridor                          │
│          │   └─ Duration: Until ambulance passes intersection               │
│          │   └─ Priority Level: 10/10 (highest)                             │
│          │                                                                  │
│  ELSE                                                                       │
│     └─ [NORMAL MODE]                                                        │
│        └─ Use optimized signal timing from RL model                         │
│        └─ Priority Level: Based on traffic density                          │
│                                                                             │
│  RESET: Emergency mode deactivates when:                                    │
│         • Vehicle no longer detected in intersection                        │
│         • Siren audio stops for >10 seconds                                 │
│         • Counter reaches timeout (120 seconds max)                         │
│                                                                             │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
                               ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│              CONTROL LAYER: SIGNAL GENERATION & COMMANDS                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Signal State Machine (Python)                                        │   │
│  │ • States: GREEN (30-60s) → YELLOW (3-5s) → RED (10-60s)              │   │
│  │ • Manages: 4 directions sequentially                                 │   │
│  │ • Cycle: Typically 60-120 seconds per full cycle                     │   │
│  │ • Phase: 2-phase (NS/EW) or 4-phase (all independent)                │   │
│  │ • Output: 12 control signals for 4 directions × 3 colors             │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                 │                                           │
│                                 ↓                                           │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Serial Protocol to Arduino (PySerial)                                │   │
│  │ • Command Format: "G1" (green on direction 1)                        │   │
│  │ •               "Y1" (yellow on direction 1)                         │   │
│  │ •               "R1" (red on direction 1)                            │   │
│  │ • Baud Rate: 9600                                                    │   │
│  │ • Port: /dev/ttyACM0 (Linux/Raspberry Pi)                            │   │
│  │ • Latency: <100ms per command                                        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                 │                                           │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
                               ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│              OUTPUT LAYER: HARDWARE ACTUATORS & VISUALIZATION                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  [HARDWARE CONTROL]                                                          │
│  • LEDs (12 total): Red/Yellow/Green for 4 directions                       │
│    └─ Controlled via Arduino digital pins (12 outputs, PWM optional)        │
│                                                                               │
│  • Buzzer (5V): Emergency alert indicator                                    │
│    └─ Triggered when ambulance detected + passing                           │
│                                                                               │
│  • 16×2 LCD Display (I2C): Show signal countdowns                           │
│    └─ Display format: "N: 15s | S: 05s | E: RED | W: YEL"                 │
│                                                                               │
│  [GUI VISUALIZATION - Tkinter Dashboard]                                    │
│  ┌────────────────────────────────────────────────────────────────┐        │
│  │ MAIN DASHBOARD TAB                                             │        │
│  │                                                                │        │
│  │ ┌─────────────────────────────────────┐                       │        │
│  │ │   LIVE VIDEO FEED (4-cam mosaic)    │  [Emergency Alert]   │        │
│  │ │   • YOLO bounding boxes overlay     │  RED BANNER          │        │
│  │ │   • Vehicle count per lane          │  "🚑 AMBULANCE      │        │
│  │ │   • Detection confidence scores     │   DETECTED!"         │        │
│  │ └─────────────────────────────────────┘                       │        │
│  │                                                                │        │
│  │ ┌─────────────────────────────────────┐                       │        │
│  │ │   SIGNAL STATUS PANEL               │                       │        │
│  │ │  North: ● 45s (GREEN)               │                       │        │
│  │ │  South: ◯ ◯ 05s (YELLOW)            │                       │        │
│  │ │  East:  ○ 30s (RED)                 │                       │        │
│  │ │  West:  ◯ ◯ 30s (RED)               │                       │        │
│  │ └─────────────────────────────────────┘                       │        │
│  │                                                                │        │
│  │ ┌─────────────────────────────────────┐                       │        │
│  │ │   REAL-TIME METRICS                 │                       │        │
│  │ │  Vehicles Passed: 156/hour          │                       │        │
│  │ │  Avg Wait Time: 25.3 sec            │                       │        │
│  │ │  Density (North): 8.2 veh/m²        │                       │        │
│  │ │  FPS: 28.5 (YOLO inference)         │                       │        │
│  │ └─────────────────────────────────────┘                       │        │
│  │                                                                │        │
│  │ [Configuration] [Analytics] [System Status]                   │        │
│  └────────────────────────────────────────────────────────────────┘        │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────┐        │
│  │ CONFIGURATION TAB (Parameter Tuning)                           │        │
│  │                                                                │        │
│  │ ML MODEL SETTINGS                                              │        │
│  │ • YOLO Confidence Threshold: [====0.75====]                  │        │
│  │ • Emergency Detection Sensitivity: [========0.85=]            │        │
│  │ • Traffic Prediction Horizon: [30s] [60s] [90s]              │        │
│  │ • Siren Detection Enabled: [✓]                               │        │
│  │                                                                │        │
│  │ SIGNAL TIMING PARAMETERS                                       │        │
│  │ • Min Green Time: [10] seconds                                │        │
│  │ • Max Green Time: [60] seconds                                │        │
│  │ • Yellow Duration: [3] seconds                                │        │
│  │ • Emergency Override Duration: [120] seconds                  │        │
│  │                                                                │        │
│  │ SENSOR CALIBRATION                                             │        │
│  │ • Ultrasonic Threshold: [100] cm                              │        │
│  │ • IR Sensor Sensitivity: [=========0.7=]                      │        │
│  │ • Sound Sensor Gain: [======0.60======]                       │        │
│  │                                                                │        │
│  │ [SAVE CONFIG] [RESET DEFAULTS] [LOAD PRESET]                 │        │
│  └────────────────────────────────────────────────────────────────┘        │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────┐        │
│  │ ANALYTICS TAB (Historical Data)                               │        │
│  │                                                                │        │
│  │ [Vehicle Count Trend - Last 24h]                             │        │
│  │ (Line chart showing vehicles/hour for each direction)        │        │
│  │                                                                │        │
│  │ [Density Heatmap - 2-hour rolling window]                    │        │
│  │ (4×4 grid showing congestion patterns)                       │        │
│  │                                                                │        │
│  │ [ML Model Performance]                                         │        │
│  │ • Detection Accuracy: 94.7%                                   │        │
│  │ • Emergency Detection Precision: 98.2%                        │        │
│  │ • Prediction Error (MAE): 1.8 vehicles                        │        │
│  │ • Average Inference Time: 65ms/frame                          │        │
│  │                                                                │        │
│  │ [Export Data] [View Reports] [Clear Logs]                    │        │
│  └────────────────────────────────────────────────────────────────┘        │
│                                                                               │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
                               ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│              DATA LOGGING & FEEDBACK LOOP                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  [SQLite Database - traffic_logs.db]                                        │
│  ├─ Table: detections                                                       │
│  │  └─ timestamp | direction | vehicle_count | density | confidence        │
│  ├─ Table: signals                                                          │
│  │  └─ timestamp | direction | signal_state | duration | mode              │
│  ├─ Table: emergencies                                                      │
│  │  └─ timestamp | type | detection_confidence | corridor_created | duration│
│  └─ Table: ml_performance                                                   │
│     └─ timestamp | model_name | accuracy | inference_time | predictions    │
│                                                                               │
│  [Feedback Loop to ML Models]                                               │
│  • Every 5 seconds: Send actual outcomes vs predictions                     │
│  • Retraining trigger: If accuracy drops below threshold (90%)              │
│  • Model update frequency: Weekly (off-peak hours)                          │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. TOTAL NUMBER OF ML MODELS REQUIRED

### 2.1 Core ML Models (5 Models)

| # | Model Name | Type | Purpose | Input | Output | Training Data |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | **YOLOv8 Vehicle Detector** | CNN Object Detection | Detect vehicles in all 4 camera feeds | RGB images (640×480) | Bounding boxes + class (car/bike/bus/truck) | Traffic Vehicles Dataset [Kaggle] - 5,000+ images |
| 2 | **Emergency Vehicle Classifier** | YOLOv8 Fine-tuned | Classify emergency vehicles from detected crops | Cropped vehicle images (80×80) | Class: ambulance/fire-truck/police/normal | Emergency Vehicle Dataset [Roboflow] - 1,000+ images |
| 3 | **Traffic Density Predictor** | LSTM Time-Series | Predict vehicle density 30-60s ahead | Historical density vectors (5min history) + time features | Predicted density [N,S,E,W] for next 60s | Custom collected data - 30+ days continuous traffic |
| 4 | **Signal Timing Optimizer** | Deep Q-Network (DQN) | Learn optimal green light duration per state | Current density + wait times + signal state | Action: green time (5-60s) per direction | Simulation + real data - 100,000+ episodes |
| 5 | **Siren Audio Detector** | CNN Audio Classifier | Detect ambulance/fire truck sirens | Audio spectrogram (frequency domain) | Binary (siren/no-siren) + confidence | Emergency vehicle siren sounds - 500+ audio clips |

### 2.2 Optional Enhancement Models (2 Models)

| # | Model Name | Type | Purpose | Priority |
| --- | --- | --- | --- | --- |
| 6 | **Vehicle Trajectory Predictor** | LSTM Regression | Predict vehicle paths for better corridor planning | Medium (Nice-to-have) |
| 7 | **Congestion Pattern Classifier** | CNN + Clustering | Classify traffic congestion types (rush hour, accident, special event) | Low (Advanced feature) |

### 2.3 Model Training Timeline

```
Week 1-2: Download & Prepare Datasets
  ├─ YOLOv8 vehicle detection dataset (Kaggle)
  ├─ Emergency vehicle images (Roboflow + YouTube data)
  └─ Siren audio clips (YouTube + emergency service recordings)

Week 3: Train Core Models
  ├─ YOLOv8 Vehicle Detector (GPU: ~4 hours)
  ├─ Emergency Vehicle Classifier (GPU: ~2 hours)
  └─ Siren Audio Detector (GPU: ~1 hour)

Week 4: Collect Custom Data & Train Prediction Models
  ├─ Collect 7+ days of traffic data using college lab hardware
  ├─ Collect emergency vehicle siren recordings
  ├─ Train Traffic Density Predictor (LSTM) (~30min)
  ├─ Train Signal Timing Optimizer (DQN) via simulation (~2-3 hours)
  └─ Validate all models on test set

Week 5: Fine-tune & Deploy to Raspberry Pi
  ├─ Convert models to TensorFlow Lite (for Raspberry Pi)
  ├─ Benchmark inference speed on Pi (target: >20 FPS)
  ├─ Optimize for edge deployment
  └─ Test on real hardware
```

**MINIMUM VIABLE PRODUCT (MVP):** Models 1-4 are mandatory.
**RECOMMENDED:** Models 1-5 for patent-worthy solution.
**ADVANCED:** Models 1-7 for production-grade system.

---

## 3. MODULE INTERACTION ARCHITECTURE

### 3.1 Module Dependency Graph

```
┌──────────────────────────────────────────────────────────────────────┐
│                     MAIN ORCHESTRATOR (main.py)                      │
│                   Manages all module interactions                     │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
         ↓                     ↓                     ↓
    ┌─────────┐          ┌─────────┐          ┌──────────┐
    │ DETECTION│         │ CONTROL │          │ GUI      │
    │ MODULE   │         │ MODULE  │          │ MODULE   │
    └─────────┘          └─────────┘          └──────────┘
         │                     │                     │
         ├─────────────────────┼─────────────────────┤
         │                     │                     │
         ↓                     ↓                     ↓
   ┌──────────┐         ┌──────────┐         ┌──────────┐
   │YOLO v8   │         │RL Model  │         │ Tkinter  │
   │Detection │         │Q-Learning│         │Dashboard │
   └──────────┘         └──────────┘         └──────────┘
         │                     │                     │
         ├─ Traffic Density    ├─ Signal State       ├─ Video Overlay
         ├─ Vehicle Count      ├─ Green Duration    ├─ Metrics Display
         ├─ Emergency Detection└─ Priority Override └─ Config Panel
         └─ Confidence Scores
```

### 3.2 Module Communication Protocol

### A. Detection Module → Control Module

```
Data Structure (JSON):
{
  "timestamp": "2026-02-08 22:45:30.123",
  "direction_data": {
    "North": {
      "vehicle_count": 12,
      "density": 8.5,
      "avg_confidence": 0.92,
      "queue_length_cm": 145
    },
    "South": { ... },
    "East": { ... },
    "West": { ... }
  },
  "emergency_detected": true,
  "emergency_type": "ambulance",
  "emergency_confidence": 0.98,
  "siren_detected": true,
  "siren_confidence": 0.89
}

Transmission: Shared Python object (same process)
Frequency: Every 1-2 seconds
```

### B. Control Module → Hardware Control (Arduino)

```
Serial Command Format:
- "G1" → Green signal Direction 1 (North)
- "Y1" → Yellow signal Direction 1
- "R1" → Red signal Direction 1
- "G2" → Green signal Direction 2 (South)
... (repeats for 3 & 4)
- "B1" → Buzzer ON
- "B0" → Buzzer OFF
- "D:15" → Display: 15 seconds (16×2 LCD)

Baud Rate: 9600
Protocol: UART (PySerial)
Latency Target: <100ms per command
```

### C. Control Module → GUI Module

```
Data Structure (Real-time metrics):
{
  "signal_states": {
    "North": {"color": "GREEN", "duration": 45},
    "South": {"color": "RED", "duration": 30},
    "East": {"color": "YELLOW", "duration": 5},
    "West": {"color": "RED", "duration": 30}
  },
  "metrics": {
    "vehicles_passed": 156,
    "avg_wait_time": 25.3,
    "mode": "NORMAL",
    "inference_fps": 28.5
  },
  "alert": {
    "active": true,
    "message": "🚑 Ambulance detected - Corridor created",
    "severity": "HIGH"
  }
}

Update Frequency: 30 FPS (33ms per update)
Connection: Shared memory / threading
```

### D. GUI Module → Control Module (Configuration Feedback)

```
Parameter Update (User interacts with Configuration Panel):
{
  "yolo_confidence": 0.75,
  "emergency_sensitivity": 0.85,
  "min_green_time": 10,
  "max_green_time": 60,
  "yellow_duration": 3,
  "siren_detection_enabled": true,
  "ultrasonic_threshold": 100
}

Update Trigger: On slider/button change
Apply Latency: <200ms
```

### 3.3 Thread & Process Architecture

```
MAIN PROCESS (Python)
│
├─ Thread 1: Camera Capture & YOLO Inference
│   ├─ OpenCV camera feeds (4 cameras)
│   ├─ YOLOv8 detection
│   ├─ Emergency vehicle classification
│   ├─ Siren audio processing
│   └─ Queue length measurement
│   └─ Cycle Time: 33-67ms (15-30 FPS per camera)
│
├─ Thread 2: ML Model Prediction & Decision
│   ├─ Traffic density prediction (LSTM)
│   ├─ Signal optimization (DQN)
│   ├─ Emergency mode logic
│   └─ Cycle Time: Every 2-5 seconds (lower frequency)
│
├─ Thread 3: Signal Control & Arduino Communication
│   ├─ State machine (signal logic)
│   ├─ Serial commands to Arduino
│   ├─ Countdown timers
│   └─ Cycle Time: Every 100-500ms
│
├─ Thread 4: GUI Dashboard Update
│   ├─ Tkinter main loop
│   ├─ Video display + overlay
│   ├─ Metrics update
│   └─ Cycle Time: 33ms (30 FPS)
│
└─ Thread 5: Data Logging & Database
    ├─ SQLite write operations
    ├─ CSV export
    ├─ Performance metrics
    └─ Cycle Time: Every 5-10 seconds (low priority)

Thread Communication: Queue (thread-safe)
Sync Points: Lock mechanisms for shared data
```

---

## 4. TECHNOLOGY STACK & HARDWARE INTEGRATION

### 4.1 Complete Technology Stack Breakdown

### LAYER 1: INPUT & SENSOR ACQUISITION

| Component | Technology | Specification | Integration Point |
| --- | --- | --- | --- |
| **Camera Feeds** | Raspberry Pi Camera Module v2 | 8MP, 1080p@30fps | OpenCV reader |
| **Backup Camera** | USB Webcam (Logitech C270) | 1280×960@30fps | OpenCV reader |
| **Vehicle Detection** | IR Obstacle Sensor (FC-51) | Range: 2-30cm | GPIO ADC (MCP3008) |
| **Queue Length** | Ultrasonic Sensor (HC-SR04) | Range: 2-400cm | Raspberry Pi GPIO |
| **Siren Detection** | Sound Sensor (KY-037) | Frequency: 20Hz-20kHz | GPIO ADC (MCP3008) |
| **Light Level** | LDR Module | 0-1024 analog | GPIO ADC (MCP3008) |
| **Time Sync** | RTC Module (DS3231) | ±2ppm accuracy | I2C (Raspberry Pi) |

### LAYER 2: PROCESSING & ML

| Component | Technology | Version | Deployment |
| --- | --- | --- | --- |
| **Language** | Python | 3.8+ | Raspberry Pi 4 |
| **CV Framework** | PyTorch / TensorFlow | 2.12+ | GPU (laptop) / CPU (Pi) |
| **Object Detection** | YOLOv8 (Ultralytics) | v8n (nano) | TFLite on Pi |
| **Time-Series** | Keras/TensorFlow | 2.12+ | TFLite on Pi |
| **RL Agent** | Python RL libraries | Stable-Baselines3 | CPU (Pi) |
| **Audio Processing** | Librosa + SciPy | 1.9+ | Real-time on Pi |
| **Data Processing** | NumPy, Pandas | Latest | Pi & development PC |

### LAYER 3: CONTROL & ACTUATION

| Component | Technology | Protocol | Specification |
| --- | --- | --- | --- |
| **Signal Controller** | Arduino Mega 2560 | UART Serial | 12 digital outputs (LEDs) |
| **Traffic LEDs** | 5mm Red/Yellow/Green | GPIO PWM | 12 total (4 directions × 3) |
| **Buzzer** | 5V Active Buzzer | GPIO ON/OFF | Emergency alert |
| **LCD Display** | 16×2 Character Display | I2C | Countdown timer display |
| **IoT Hub** | ESP32 DevKit | WiFi/MQTT | Optional cloud logging |

### LAYER 4: VISUALIZATION & INTERFACE

| Component | Technology | Libraries | Features |
| --- | --- | --- | --- |
| **GUI Framework** | Tkinter | ttkbootstrap | Cross-platform |
| **Video Display** | OpenCV + PIL | cv2 + Pillow | Live camera feed |
| **Graphics** | Matplotlib | 3.5+ | Real-time charts |
| **Database** | SQLite | sqlite3 | Local data persistence |
| **Webserver** | Flask (optional) | 2.2+ | API endpoints |

### LAYER 5: SIMULATION & TESTING

| Component | Technology | Purpose | Usage |
| --- | --- | --- | --- |
| **Arduino Sim** | Proteus | Circuit simulation | Test LED control without hardware |
| **Traffic Sim** | SUMO (optional) | Generate synthetic traffic | Test algorithm on various scenarios |
| **Model Training** | Google Colab | Free GPU | Train YOLOv8 & LSTM models |

### 4.2 Step-by-Step Tech Stack Integration Timeline

```
MONTH 1: SETUP & DATA PREPARATION
├─ Week 1: Environment Setup
│  ├─ Install Python 3.8+ on Raspberry Pi 4
│  │  Commands:
│  │  $ sudo apt update && sudo apt upgrade
│  │  $ sudo apt install python3-pip python3-dev
│  │  $ pip install numpy pandas opencv-python
│  │
│  ├─ Install PyTorch/TensorFlow
│  │  $ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
│  │  $ pip install tensorflow
│  │
│  ├─ Install Ultralytics YOLO
│  │  $ pip install ultralytics
│  │
│  ├─ Install Arduino IDE & setup Proteus
│  │  ├─ Download Arduino IDE from arduino.cc
│  │  ├─ Download Proteus 8.12+
│  │  └─ Load example Arduino code (traffic light control)
│  │
│  └─ Setup GPIO & Serial Libraries
│     $ pip install RPi.GPIO pyserial
│
├─ Week 2: Dataset Collection & Labeling
│  ├─ Download from Kaggle:
│  │  ├─ Traffic Vehicles Detection Dataset
│  │  ├─ Vehicle Detection Image Dataset
│  │  └─ Emergency Vehicle Dataset
│  │
│  ├─ Collect custom data using college lab setup
│  │  ├─ Record 7+ days of traffic videos (parking, corridor)
│  │  ├─ Record siren audio clips
│  │  └─ Label with Roboflow GUI
│  │
│  └─ Prepare dataset structure:
│     data/
│     ├─ images/
│     │  ├─ train/
│     │  └─ val/
│     └─ labels/
│        ├─ train/
│        └─ val/
│
├─ Week 3: Model Training on Laptop/Colab
│  ├─ Train YOLOv8 Vehicle Detector
│  │  Python code:
│  │  from ultralytics import YOLO
│  │  model = YOLO('yolov8n.pt')
│  │  results = model.train(data='data.yaml', epochs=100, imgsz=640)
│  │  model.export(format='tflite')  # For Raspberry Pi
│  │
│  ├─ Fine-tune Emergency Vehicle Classifier
│  │  model = YOLO('yolov8n.pt')
│  │  results = model.train(data='emergency_vehicles.yaml', epochs=50)
│  │  model.export(format='tflite')
│  │
│  └─ Train Siren Detector (Audio CNN)
│     • Preprocess audio → spectrogram
│     • CNN model (Librosa + TensorFlow)
│     • Export as SavedModel format
│
└─ Week 4: Prepare Models for Edge Deployment
   ├─ Convert YOLOv8 to TensorFlow Lite
   ├─ Quantize models (int8 for 4x speedup)
   ├─ Test inference speed on Raspberry Pi
   └─ Target: >15 FPS per camera
```

```
MONTH 2: HARDWARE & SOFTWARE INTEGRATION
├─ Week 5: Arduino & Proteus Setup
│  ├─ Write Arduino code for LED control
│  │  // Arduino Mega sketch
│  │  #define RED_N 22, YELLOW_N 24, GREEN_N 26
│  │  #define RED_S 28, YELLOW_S 30, GREEN_S 32
│  │  #define RED_E 34, YELLOW_E 36, GREEN_E 38
│  │  #define RED_W 40, YELLOW_W 42, GREEN_W 44
│  │  #define BUZZER 46
│  │
│  │  void setup() {
│  │    Serial.begin(9600);
│  │    for(int i=22; i<=46; i+=2) pinMode(i, OUTPUT);
│  │  }
│  │
│  │  void loop() {
│  │    if(Serial.available()) {
│  │      String cmd = Serial.readString();
│  │      if(cmd == "G1") digitalWrite(GREEN_N, HIGH);  // Green North
│  │      else if(cmd == "R1") digitalWrite(RED_N, HIGH);  // Red North
│  │      // ... handle other signals
│  │    }
│  │  }
│  │
│  ├─ Test in Proteus simulation
│  │  ├─ Create circuit: Arduino Mega + 12 LEDs + resistors
│  │  ├─ Use Virtual Serial Port (VSP) for Python communication
│  │  └─ Verify LED switching works via Python commands
│  │
│  └─ Upload to physical Arduino Mega
│     $ arduino-cli upload --fqbn arduino:avr:mega2560 --port /dev/ttyACM0
│
├─ Week 6: Sensor Integration on Raspberry Pi
│  ├─ Setup IR Sensors (FC-51)
│  │  Pin: GPIO 17 (Analog via MCP3008)
│  │  Calibration: Test at different distances
│  │
│  ├─ Setup Ultrasonic Sensors (HC-SR04) - 4 units
│  │  Pins: GPIO 17(trigger), GPIO 27(echo) per sensor
│  │  Code:
│  │  import RPi.GPIO as GPIO
│  │  import time
│  │  GPIO.setup(17, GPIO.OUT)  # Trigger
│  │  GPIO.setup(27, GPIO.IN)    # Echo
│  │  # Measure distance using timing
│  │
│  ├─ Setup Sound Sensor (KY-037)
│  │  Pin: GPIO ADC (MCP3008 channel 0)
│  │  Calibration: Record baseline vs siren noise
│  │
│  ├─ Setup RTC (DS3231)
│  │  Protocol: I2C (pins GPIO 2, 3)
│  │  Sync: $ sudo hwclock --systohc
│  │
│  └─ Verify all sensors output correct values
│     Create test script: sensor_calibration.py
│
├─ Week 7: OpenCV & Camera Setup
│  ├─ Connect Raspberry Pi Camera Module v2
│  │  CSI ribbon cable to Pi camera port
│  │
│  ├─ Test video capture
│  │  import cv2
│  │  cap = cv2.VideoCapture(0)
│  │  ret, frame = cap.read()
│  │  print(f"Resolution: {frame.shape}")  # Should be (480, 640, 3)
│  │
│  ├─ Setup USB Webcam (alternate)
│  │  import cv2
│  │  cap = cv2.VideoCapture(0)  # or /dev/video1 if conflict
│  │
│  └─ Calibrate camera exposure for day/night
│     Use LDR sensor to adjust brightness
│
├─ Week 8: Detection Module Development
│  ├─ Load TFLite models on Pi
│  │  import tensorflow as tf
│  │  interpreter = tf.lite.Interpreter(model_path='yolov8n.tflite')
│  │  interpreter.allocate_tensors()
│  │
│  ├─ Run inference on camera frames
│  │  input_details = interpreter.get_input_details()
│  │  output_details = interpreter.get_output_details()
│  │  # Preprocess frame → model → get detections
│  │
│  ├─ Extract vehicle count per lane
│  │  • Define lane zones (ROI) in image
│  │  • Count detections per zone
│  │  • Calculate density = count / zone_area
│  │
│  └─ Validate with IR sensors
│     Create validation script: compare_detection_methods.py
│
└─ Week 9: Control & Arduino Integration
   ├─ Establish serial communication
   │  import serial
   │  ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
   │  ser.write(b'G1')  # Send command
   │
   ├─ Implement signal state machine
   │  class TrafficLight:
   │    def __init__(self):
   │      self.state = 'RED'  # RED, YELLOW, GREEN
   │      self.duration = 30
   │      self.start_time = time.time()
   │
   │    def update(self, new_state, new_duration):
   │      self.state = new_state
   │      self.duration = new_duration
   │      ser.write(f'G{dir}'.encode())  # Send to Arduino
   │
   ├─ Test full loop: detection → decision → Arduino → LEDs
   │  1. Capture frame from camera
   │  2. Run YOLO inference
   │  3. Calculate traffic density
   │  4. Call signal controller
   │  5. Send command to Arduino
   │  6. Verify LED changes
   │
   └─ Measure latency: end-to-end <500ms target
```

```
MONTH 3: ML MODELS & GUI
├─ Week 10: RL Signal Optimization Training
│  ├─ Build simulation environment
│  │  class TrafficEnv:
│  │    def __init__(self):
│  │      self.lanes = {'N': 0, 'S': 0, 'E': 0, 'W': 0}
│  │      self.signal_state = 'N_GREEN'  # 8 possible states
│  │
│  │    def step(self, action):  # action = green_duration (5-60s)
│  │      # Update vehicle count based on duration
│  │      # Calculate reward = throughput - wait_time
│  │      return observation, reward, done, info
│  │
│  ├─ Train DQN agent
│  │  from stable_baselines3 import DQN
│  │  model = DQN('MlpPolicy', env, learning_rate=1e-3)
│  │  model.learn(total_timesteps=100000)
│  │  model.save('signal_optimizer.zip')
│  │
│  └─ Export trained model for Pi
│
├─ Week 11: LSTM Traffic Prediction Training
│  ├─ Prepare time-series data
│  │  • Collect 30+ days of vehicle count data
│  │  • Features: density, time_of_day, day_of_week, weather
│  │  • Normalize: (x - mean) / std
│  │
│  ├─ Build LSTM model
│  │  model = Sequential([
│  │    LSTM(128, return_sequences=True, input_shape=(60, 4)),
│  │    Dropout(0.2),
│  │    LSTM(64),
│  │    Dense(32, activation='relu'),
│  │    Dense(4)  # Output: [N_density, S_density, E_density, W_density]
│  │  ])
│  │  model.compile(optimizer='adam', loss='mse')
│  │  history = model.fit(X_train, y_train, epochs=50, validation_split=0.2)
│  │
│  └─ Save as TFLite for Pi
│     converter = tf.lite.TFLiteConverter.from_keras_model(model)
│     tflite_model = converter.convert()
│
├─ Week 12: Tkinter GUI Development
│  ├─ Create main window structure
│  │  root = tk.Tk()
│  │  root.title("Adaptive Traffic Management System")
│  │  root.geometry("1280x720")
│  │
│  ├─ Implement tabs: Dashboard, Config, Analytics
│  │  notebook = ttk.Notebook(root)
│  │  tab1 = ttk.Frame(notebook)  # Dashboard
│  │  tab2 = ttk.Frame(notebook)  # Configuration
│  │  tab3 = ttk.Frame(notebook)  # Analytics
│  │  notebook.add(tab1, text="Dashboard")
│  │  notebook.add(tab2, text="Configuration")
│  │  notebook.add(tab3, text="Analytics")
│  │
│  ├─ Dashboard Tab:
│  │  • 2×2 grid for 4 camera feeds (OpenCV + PIL)
│  │  • YOLO bounding boxes overlay
│  │  • Signal status (colored circles: Red/Yellow/Green)
│  │  • Metric displays (wait time, throughput, FPS)
│  │  • Emergency alert banner (red background, flashing)
│  │
│  ├─ Configuration Tab:
│  │  • Sliders: YOLO confidence (0.3-0.9)
│  │  • Sliders: Min/max green time, yellow duration
│  │  • Checkboxes: Enable siren detection, emergency mode
│  │  • Buttons: Save config, Reset defaults, Load preset
│  │
│  ├─ Analytics Tab:
│  │  • Matplotlib figures embedded in Tkinter
│  │  • Vehicle count trend (24h)
│  │  • Density heatmap
│  │  • Model performance metrics
│  │  • Buttons: Export data, View reports
│  │
│  └─ Real-time update every 33ms (30 FPS)
│     def update_gui():
│       # Get latest data from control module
│       # Update video frames
│       # Update metric labels
│       # Redraw charts
│       root.after(33, update_gui)
│
└─ Week 13-14: Integration Testing & Optimization
   ├─ End-to-end testing
   │  1. Start all threads
   │  2. Verify camera capture works
   │  3. Run YOLO on live frames
   │  4. Check sensor readings
   │  5. Verify Arduino commands
   │  6. Confirm GUI updates
   │  7. Monitor for crashes/bottlenecks
   │
   ├─ Performance optimization
   │  • Profile Python code with cProfile
   │  • Identify bottlenecks
   │  • Optimize YOLO with TFLite Delegate
   │  • Target: 25 FPS with all modules running
   │
   └─ Stress testing
      • Run for 8+ hours continuously
      • Monitor memory/CPU usage
      • Check for memory leaks
      • Verify database integrity
```

### 4.3 Complete Hardware Wiring Diagram (Text-Based)

```
RASPBERRY PI 4 (8GB)
├─ Power: 5V 3A USB-C
├─ Camera: CSI Ribbon to Camera Port
├─ GPIO Pins:
│  ├─ GPIO 17 → MCP3008 CH0 (IR Sensor)
│  ├─ GPIO 27 → MCP3008 CH1 (Sound Sensor)
│  ├─ GPIO 22 → Ultrasonic1 Trigger
│  ├─ GPIO 23 → Ultrasonic1 Echo
│  ├─ GPIO 24 → Ultrasonic2 Trigger
│  ├─ GPIO 25 → Ultrasonic2 Echo
│  ├─ GPIO 26 → Ultrasonic3 Trigger
│  ├─ GPIO 27 → Ultrasonic3 Echo (via 1K resistor voltage divider)
│  └─ GPIO 2 (SDA) & GPIO 3 (SCL) → I2C (RTC DS3231, LCD 16×2)
│
├─ UART Serial:
│  ├─ TX (GPIO 14) → Arduino RX0
│  └─ RX (GPIO 15) → Arduino TX0
│
└─ USB Ports:
   ├─ USB Port 1 → Logitech Webcam
   ├─ USB Port 2 → USB-to-Serial (optional backup)
   └─ USB Port 3 → External SSD/HDD (data logging)

ARDUINO MEGA 2560
├─ Power: 9V 1A adapter
├─ Serial RX/TX:
│  ├─ RX0 → Raspberry Pi TX
│  └─ TX0 → Raspberry Pi RX
├─ Digital Outputs (LED Control):
│  ├─ Pin 22 → 220Ω Resistor → Red LED (North)
│  ├─ Pin 24 → 220Ω Resistor → Yellow LED (North)
│  ├─ Pin 26 → 220Ω Resistor → Green LED (North)
│  ├─ Pin 28 → 220Ω Resistor → Red LED (South)
│  ├─ Pin 30 → 220Ω Resistor → Yellow LED (South)
│  ├─ Pin 32 → 220Ω Resistor → Green LED (South)
│  ├─ Pin 34 → 220Ω Resistor → Red LED (East)
│  ├─ Pin 36 → 220Ω Resistor → Yellow LED (East)
│  ├─ Pin 38 → 220Ω Resistor → Green LED (East)
│  ├─ Pin 40 → 220Ω Resistor → Red LED (West)
│  ├─ Pin 42 → 220Ω Resistor → Yellow LED (West)
│  ├─ Pin 44 → 220Ω Resistor → Green LED (West)
│  └─ Pin 46 → 5V Buzzer (cathode to GND)
└─ All LED cathodes → GND

SENSORS (Connected to Raspberry Pi)
├─ IR Obstacle Sensor (FC-51) × 4
│  ├─ VCC → 3.3V
│  ├─ GND → GND
│  └─ OUT → MCP3008 ADC (Channels 0-3)
│
├─ Ultrasonic Sensor (HC-SR04) × 4
│  ├─ VCC → 5V (regulated)
│  ├─ GND → GND
│  ├─ TRIG → GPIO (22, 24, 26, X)
│  └─ ECHO → GPIO via 1K/2K voltage divider (20, 21, X, X)
│     Voltage divider: 5V signal → 3.3V safe for Pi GPIO
│
├─ Sound Sensor (KY-037) × 1
│  ├─ VCC → 3.3V
│  ├─ GND → GND
│  ├─ A0 → MCP3008 Ch1 (analog audio)
│  └─ D0 → GPIO 13 (digital threshold, optional)
│
├─ LDR Module (Light Sensor)
│  ├─ VCC → 3.3V
│  ├─ GND → GND
│  └─ OUT → MCP3008 Ch2
│
└─ RTC Module (DS3231)
   ├─ VCC → 3.3V
   ├─ GND → GND
   ├─ SDA → GPIO 2
   └─ SCL → GPIO 3

DISPLAY (Connected to Raspberry Pi via I2C)
├─ 16×2 Character LCD (I2C backpack)
│  ├─ VCC → 5V
│  ├─ GND → GND
│  ├─ SDA → GPIO 2
│  └─ SCL → GPIO 3
│
└─ Displays: Signal state + countdown timer

PROTEUS SIMULATION (Virtual Circuit)
├─ Virtual Serial Port (COM3 ↔ /dev/ttyACM0)
├─ Arduino Mega 2560 (Simulated)
├─ 12× LED indicators (Red/Yellow/Green)
└─ Breadboard connections matching real setup

LED CONNECTIONS (Detailed)
├─ North Direction:
│  ├─ Red LED: Arduino Pin 22 → 220Ω resistor → LED → GND
│  ├─ Yellow LED: Arduino Pin 24 → 220Ω resistor → LED → GND
│  └─ Green LED: Arduino Pin 26 → 220Ω resistor → LED → GND
│
├─ South Direction: Pins 28, 30, 32
├─ East Direction: Pins 34, 36, 38
└─ West Direction: Pins 40, 42, 44

POWER DISTRIBUTION
├─ Raspberry Pi 4: 5V 3A via USB-C
├─ Arduino Mega: 9V 1A via barrel jack
├─ Sensors: 3.3V for digital, 5V (regulated) for ultrasonic
├─ LEDs: +5V from Arduino PWM pins (max 40mA total current limit)
└─ Buzzer: 5V from Arduino pin 46

PROTEUS SIMULATION SETUP
1. Create circuit in Proteus:
   - Drag Arduino Mega 2560 module
   - Add 12 LEDs (LED red/yellow/green)
   - Add Virtual Serial Port (VSP)
   - Connect Arduino digital pins to LED cathodes
   - All LED anodes to +5V
   - VSP to Arduino RX/TX pins

2. Upload Arduino hex file:
   - Compile in Arduino IDE
   - Copy .hex file to Proteus project folder

3. Configure VSP:
   - Proteus: Instruments → Virtual Serial Port
   - Set COM port (e.g., COM3)
   - Baud rate: 9600

4. Run simulation:
   - Start Proteus simulation
   - Python sends commands: ser.write(b'G1')
   - LEDs toggle in Proteus in real-time
```

### 4.4 Detailed Tech Stack Table (All Components)

| Layer | Component | Tech | Language | Library | Hardware | Pins/Port |
| --- | --- | --- | --- | --- | --- | --- |
| **Input** | Camera 1-4 | OpenCV | Python | cv2.VideoCapture() | Pi Camera v2 | CSI ribbon |
|  | IR Sensor × 4 | ADC | Python | RPi.GPIO + MCP3008 | FC-51 | GPIO 17 |
|  | Ultrasonic × 4 | GPIO PWM | Python | RPi.GPIO | HC-SR04 | GPIO 22-27 |
|  | Sound Sensor | ADC | Python | MCP3008 + SciPy | KY-037 | GPIO 27 |
|  | RTC | I2C | Python | smbus2 | DS3231 | GPIO 2,3 |
| **Process** | YOLO Detection | ML | Python | ultralytics | CPU/GPU | - |
|  | Emergency Classifier | ML | Python | TensorFlow Lite | Pi CPU | - |
|  | LSTM Predictor | ML | Python | TensorFlow Lite | Pi CPU | - |
|  | DQN Optimizer | ML | Python | Stable-Baselines3 | Pi CPU | - |
|  | Audio Detector | ML | Python | Librosa + TensorFlow | Pi CPU | - |
| **Control** | Signal FSM | Logic | Python | Custom module | - | - |
|  | Arduino Interface | Serial | Python | PySerial | UART | GPIO 14,15 |
|  | Database | SQL | Python | sqlite3 | Pi storage | /home/pi/data/ |
| **Output** | LEDs × 12 | GPIO | Arduino C | digitalWrite() | LED RGB | Pins 22-44 |
|  | Buzzer | GPIO | Arduino C | digitalWrite() | 5V buzzer | Pin 46 |
|  | LCD Display | I2C | Python | smbus2 + custom driver | 16×2 LCD | GPIO 2,3 |
| **Visualization** | GUI Dashboard | UI | Python | tkinter + ttkbootstrap | Monitor | HDMI |
|  | Video Overlay | Graphics | Python | PIL + OpenCV | Pi display | HDMI/X11 |
|  | Real-time Charts | Plotting | Python | matplotlib | Pi display | - |
| **Testing** | Arduino Simulation | Simulation | C | Arduino IDE | Virtual | COM3 |
|  | Circuit Design | CAD | - | Proteus 8.12 | Virtual | VSP |
|  | Model Training | ML Ops | Python | Jupyter + Colab | GPU (cloud) | - |

---

## 5. DATA FLOW WITH TIMESTAMPS & LATENCY BUDGET

```
FRAME CAPTURE: 0ms
├─ Pi Camera capture
└─ Timestamp: T0

YOLO INFERENCE: 0-67ms (target: 33ms per frame)
├─ Pre-process: Resize to 640×480, normalize
├─ Forward pass: 30-50ms
├─ Post-process: NMS, confidence filtering
└─ Output: Bounding boxes + scores at T0+33ms

EMERGENCY CLASSIFICATION: 33-50ms
├─ Crop detected vehicle regions
├─ Run fine-tuned classifier
└─ Output: Class confidence at T0+50ms

DENSITY CALCULATION: 50-65ms
├─ Count detections per lane ROI
├─ Query ultrasonic queue length
├─ Calculate density vector
└─ Output at T0+65ms

DECISION MAKING: 65-100ms (every 2-5 seconds)
├─ Check emergency flag
├─ Query LSTM predictor (if interval hit)
├─ Query DQN for next action
└─ Output: Signal state at T0+100ms

ARDUINO COMMAND: 100-150ms
├─ Format serial command
├─ Transmit via UART (9600 baud)
├─ Arduino receives + parses
└─ LED switches at T0+150ms (target: <200ms total)

GUI UPDATE: Every 33ms in parallel thread
└─ Display delay: +16ms (60Hz monitor refresh rate)

TOTAL LATENCY: 200ms (acceptable for traffic signals)
```

---

This completes the comprehensive bird’s-eye view of your Adaptive Traffic Signal system. All components, models, interactions, and technologies are clearly mapped with specific implementation details for your college lab prototype.
