# ZENTRA — Multi-Camera Edge Architecture Review & Deep-Dive Requirements

> **⚠️ เอกสารเก็บไว้อ้างอิงเท่านั้น — ไม่ใช่สภาพระบบปัจจุบัน**
>
> นี่คือ brief ที่เขียน *ก่อน* ลงมือ refactor multi-camera งานตามเอกสารนี้ทำเสร็จแล้ว
> เก็บไว้เพื่อให้เห็นเหตุผลเบื้องหลังการออกแบบ (ทำไมถึงเลือกแบบนี้)
>
> ส่วนที่ **ไม่ตรงกับของจริงแล้ว**:
> - ระบบ multi-view fusion / เทียบพื้นที่ (ground-plane calibration) ถูกถอดออกทั้งหมด
> - เมนูประมวลผลบน Cloud ถูกถอดออก (โค้ดใน `cloud/` ยังอยู่ แต่ไม่มีทางเข้าจาก UI)
>
> **ต้องการรู้ว่าตอนนี้ระบบเป็นยังไง → อ่าน [STRUCTURE.md](STRUCTURE.md)**

## Purpose

เอกสารนี้ใช้เป็น technical brief สำหรับให้ Claude Code ตรวจสอบและปรับปรุงสถาปัตยกรรม Multi-Camera ของ ZENTRA ก่อนเริ่ม refactor จริง

เป้าหมายหลัก:

- รองรับ CCTV หลายกล้องบน Edge Device
- แต่ละกล้องกำหนด `role` / `ppe_items` / `zone` / detection requirements ได้อิสระ
- ใช้โมเดล AI ชุดเดียวร่วมกัน ไม่โหลด `.pt` ซ้ำต่อกล้อง
- Batch inference ข้ามกล้อง
- ควบคุม GPU/CPU budget และลด FPS ต่อกล้องอย่าง graceful เมื่อจำนวนกล้องเพิ่ม
- รักษา behavior เดิมของ PPEEngine ให้ได้มากที่สุด
- ไม่ให้ระบบสะสม frame เก่าจนเกิด latency สูง
- รองรับ RTX 3050 4GB และสามารถขยายไป TensorRT FP16 / Cloud ModelHub ในอนาคต

---

# 1. Current Problem

สถาปัตยกรรมเดิมของ `MultiCameraManager` สร้าง `Pipeline + PPEEngine` เต็มชุดต่อกล้อง

แต่ละ `PPEEngine` โหลดโมเดล เช่น:

- Person detector (`yolo11m`)
- PPE detector (`ppe_finetuned`)
- Pose (`yolo11n-pose`)
- Fall TFLite

เมื่อมี N กล้อง:

```text
VRAM ≈ ModelMemory × N
Compute ≈ InferenceCost × N
```

ทำให้ RTX 3050 4GB รองรับจำนวนกล้องจำกัด และ compute ไม่สามารถแชร์ได้อย่างมีประสิทธิภาพ

---

# 2. Proposed High-Level Architecture

แนวคิดหลัก:

> แยก Shared AI Models + GPU Scheduling ออกจาก Per-Camera State

```text
Cam 0 ─┐
Cam 1 ─┤
Cam 2 ─┤
Cam N ─┘
   │
   ▼
Capture Manager
   │
   ▼
Latest Frame Buffer
   │
   ▼
Inference Scheduler
   │
   ├── Person Batch
   ├── PPE Batch
   ├── Pose Batch
   └── Fall Tasks
   │
   ▼
Shared ModelHub
   │
   ▼
Result Router
   │
   ├── CameraEngine 0
   ├── CameraEngine 1
   ├── CameraEngine 2
   └── CameraEngine N
          │
          ├── ByteTracker
          ├── PPE Association
          ├── Temporal Confirmers
          ├── Zones
          ├── Roles/PPE Config
          ├── Fall State
          └── Alert / Event State
```

---

# 3. Important Architectural Principle

## AI Result ไม่ควรถูกผูกกับ Frame แบบ implicit

ระบบควรมี explicit data identity:

```python
FramePacket(
    camera_id,
    frame_id,
    timestamp,
    image
)
```

และ:

```python
InferenceResult(
    camera_id,
    frame_id,
    timestamp,
    detections,
    model_type
)
```

ห้ามพึ่งพาเพียงลำดับการเรียก function หรือ shared mutable state เพื่อระบุว่า result เป็นของกล้อง/เฟรมใด

---

# 4. Component Responsibilities

## 4.1 Capture Manager

รับผิดชอบ:

- RTSP / video source
- decode
- capture loop
- latest-frame storage
- reconnect
- frame_id
- timestamp

ไม่ควรรับผิดชอบ:

- AI inference
- tracking
- PPE logic
- alert logic

### Important

ใช้ Latest Frame Buffer หรือ bounded queue ที่มี capacity ต่ำ

ไม่ควรปล่อย unbounded queue

เหตุผล:

หากกล้องส่ง 30 FPS แต่ AI ทำได้ 10 FPS:

```text
Bad:

Frame 1
Frame 2
Frame 3
...
Frame 100
```

จะเกิด backlog และ inference lag

ต้องยอม drop frame เก่าเพื่อรักษาความสด:

```text
Frame 101
Frame 102
Frame 103

AI ยังไม่อ่าน 101
→ overwrite
→ process 103
```

สำหรับ CCTV safety monitoring:

> Freshness สำคัญกว่าการประมวลผลทุก frame

---

# 5. Capture FPS / Inference FPS / Display FPS Must Be Independent

อย่าใช้ตัวแปร FPS เดียว

ระบบควรแยก:

```text
Capture FPS
Inference FPS
Display FPS
Event FPS
```

ตัวอย่าง:

```text
Camera source: 30 FPS

Capture:    30 FPS
Inference:   8 FPS
Display:    30 FPS
Event:      event-driven
```

Display สามารถแสดง frame ล่าสุดต่อเนื่องได้ แม้ inference จะทำช้ากว่า

---

# 6. ModelHub

ไฟล์:

```text
backend/utils/model_hub.py
```

หน้าที่:

- โหลด shared models เพียงครั้งเดียว
- เป็น stateless inference layer
- ไม่ถือ camera-specific state
- ไม่สร้าง track ID
- ไม่ทำ temporal confirmation
- ไม่รู้จัก zone หรือ role ของกล้อง

ตัวอย่าง API:

```python
detect_persons(frames: list[FramePacket]) -> list[InferenceResult]

detect_items(frames: list[FramePacket]) -> list[InferenceResult]

pose(frames: list[FramePacket]) -> list[InferenceResult]
```

ถ้ามี fall model ที่ยังไม่เหมาะกับ batch ให้แยกเป็น task/service ภายใน ModelHub หรือ FallInferenceWorker

---

# 7. ModelHub Must Be Shared, But Models Should Be Independently Scheduled

อย่าผูก pipeline แบบ:

```python
process_camera():
    person()
    ppe()
    pose()
    fall()
```

ทุกครั้ง

เพราะกล้องแต่ละตัวมี requirement ต่างกัน

ควรคิดเป็น model task:

```text
Person
  target FPS = 10

PPE
  target FPS = 5

Pose
  target FPS = 3

Fall
  target FPS = 2
```

และแต่ละ camera ระบุ requirements:

```python
camera.requirements = {
    "person": True,
    "ppe": True,
    "pose": False,
    "fall": False
}
```

Scheduler ต้องไม่ส่ง task ที่ไม่จำเป็นเข้า GPU

---

# 8. CameraEngine

Refactor จาก `PPEEngine`

ไฟล์อาจเป็น:

```text
backend/utils/camera_engine.py
```

หรือชื่ออื่นตาม project convention

CameraEngine ต้องถือเฉพาะ state ต่อกล้อง:

```text
CameraEngine
├── TrackerState
├── TemporalState
├── ZoneState
├── AlertState
├── FallState
└── CameraConfig
```

ไม่ควรถือ:

- YOLO model
- Pose model
- PPE model
- global GPU resource

---

# 9. CameraEngine Responsibilities

Reuse logic เดิมจาก `PPEEngine`:

- `ByteTracker`
- PPE association
- temporal confirmation
- zones
- roles
- PPE item configuration
- alert cooldown
- de-dup
- fall state
- drawing
- `_rec_style`
- `draw_items`
- existing event behavior

หลักสำคัญ:

> เปลี่ยนแหล่งข้อมูลจาก "เรียก model เอง" เป็น "รับ inference result จาก ModelHub"

---

# 10. Tracker Isolation

แต่ละกล้องต้องมี ByteTracker instance ของตัวเอง

```text
Camera 0 → Tracker 0
Camera 1 → Tracker 1
Camera 2 → Tracker 2
```

ห้ามแชร์ tracker instance ระหว่างกล้อง

เพราะ track ID และ temporal state ต้องไม่ปนกัน

---

# 11. Tracker Parity Risk

Standalone `ByteTracker` ใน:

```text
backend/utils/tracker.py
```

เป็น IoU-based implementation และอาจ behavior ต่างจาก Ultralytics native ByteTrack

ต้อง validate:

- track ID continuity
- ID switching
- occlusion
- crossing
- entry/exit
- track buffer
- matching threshold
- PPE state continuity

โดยเฉพาะ:

```text
Person ID 17
helmet confirmed

ID switch

Person ID 18
helmet unknown
```

อาจทำให้ temporal logic และ alert behavior เปลี่ยน

ดังนั้นต้องทำ parity test ก่อนตัด native tracker ออกถาวร

---

# 12. Timestamp and Frame ID

ทุก frame ต้องมี:

```text
camera_id
frame_id
timestamp
```

ทุก inference result ต้อง retain:

```text
camera_id
frame_id
timestamp
```

เพื่อป้องกัน:

- stale results
- result routing ผิดกล้อง
- temporal state ผิดลำดับ
- asynchronous inference race
- delayed result ถูกใช้กับ frame ใหม่

---

# 13. Result Router

ต้องมี explicit routing:

```text
Batch Result
    │
    ├── camera_id = 0 → CameraEngine 0
    ├── camera_id = 1 → CameraEngine 1
    └── camera_id = N → CameraEngine N
```

อย่า assume ว่า list index เท่ากับ camera ID เสมอ

ใช้ metadata ของ FramePacket/InferenceResult เป็น source of truth

---

# 14. InferenceScheduler

ไฟล์:

```text
pipeline/scheduler.py
```

เป็นหัวใจของระบบ

หน้าที่:

1. อ่าน latest frame ของ active cameras
2. ตรวจสอบ model requirements
3. ตรวจ target FPS
4. ตรวจ freshness
5. จัดลำดับ task
6. รวม frame เป็น batch
7. เรียก ModelHub
8. route results กลับ CameraEngine
9. monitor latency / dropped tasks
10. ปรับ budget เมื่อ GPU workload สูง

---

# 15. Do Not Use Simple Round-Robin as Final Scheduler

Round-robin สามารถใช้เป็น baseline ได้:

```text
Cam0 → Cam1 → Cam2 → Cam3
```

แต่ไม่ควรเป็น final design

เพราะ workload แต่ละกล้องต่างกัน:

```text
Cam A: Person only
Cam B: Person + Helmet
Cam C: Person + Pose + Fall
Cam D: Person only
```

ควรพิจารณา:

```text
priority
target_fps
last_inference_time
frame_age
model_requirement
estimated_cost
queue age
camera activity
```

---

# 16. Frame Freshness Must Affect Scheduling

Scheduler ต้องไม่เพียงถาม:

> "กล้องนี้ถึงเวลารันหรือยัง?"

แต่ต้องถาม:

> "frame ล่าสุดของกล้องนี้เก่าแค่ไหน?"

ตัวอย่าง:

```text
Camera A
target = 10 FPS
last inference = 80ms ago

Camera B
target = 10 FPS
last inference = 400ms ago
```

ควร prioritize Camera B

เพราะ B มี stale result มากกว่า

---

# 17. Per-Model Scheduling

แนะนำให้ Scheduler มอง workload เป็น task:

```text
Camera 0 + Person
Camera 0 + PPE

Camera 1 + Person

Camera 2 + Person
Camera 2 + Pose
Camera 2 + Fall
```

ไม่ใช่:

```text
Camera 0 = one giant inference task
```

สิ่งนี้ทำให้ลด compute ได้มากเมื่อกล้องมี role ต่างกัน

---

# 18. Fall Detection Optimization

Fall detection อาจเป็น bottleneck

หาก implementation เดิมใช้:

```text
30 frames / person
```

ไม่ควร run fall model กับทุก person ทุก frame

ควรมี candidate gating:

```text
Person Track
     ↓
Cheap motion / pose heuristic
     ↓
Possible fall?
     ↓ yes
Fall inference
```

หรืออย่างน้อย throttle:

```text
Fall target FPS = 1–2 FPS
```

ต่อกล้อง/ต่อ candidate ตาม benchmark

---

# 19. Adaptive Graceful Degradation

ระบบไม่ควร crash เมื่อเพิ่มกล้อง

ควรลด quality/AI frequency ตาม priority

ตัวอย่าง:

### Level 0

```text
Person 10 FPS
PPE 10 FPS
Pose 5 FPS
Fall 5 FPS
```

### Level 1

```text
Person 10 FPS
PPE 7 FPS
Pose 3 FPS
Fall 3 FPS
```

### Level 2

```text
Person 8 FPS
PPE 5 FPS
Pose 2 FPS
Fall 2 FPS
```

### Level 3

```text
Person 5 FPS
PPE 3 FPS
Pose OFF
Fall 1 FPS
```

### Level 4

```text
Person 3 FPS
PPE 2 FPS
Pose OFF
Fall OFF
```

Display ยังทำงานตาม source FPS ได้

---

# 20. Resource Manager

ควรพิจารณาเพิ่ม component:

```text
ResourceManager
```

ติดตาม:

```text
GPU utilization
VRAM
CPU utilization
RAM
inference latency
batch size
dropped tasks
frame age
```

แล้วส่งข้อมูลให้ Scheduler

เป้าหมายคือ adaptive resource allocation

---

# 21. Batch Size Optimization

อย่าคิดว่า batch ใหญ่สุด = เร็วสุด

ต้อง benchmark:

```text
Batch 1
Batch 2
Batch 4
Batch 8
Batch 16
```

และวัด:

- inference latency
- throughput
- VRAM
- GPU utilization
- end-to-end latency

หา optimal batch size สำหรับ RTX 3050 4GB

---

# 22. CPU / Decode / Memory Transfer Must Be Benchmarked

อย่าสรุปว่า GPU inference เป็น bottleneck โดยอัตโนมัติ

Pipeline จริง:

```text
RTSP
 ↓
Decode
 ↓
Resize
 ↓
Color conversion
 ↓
CPU → GPU
 ↓
Inference
 ↓
GPU → CPU
 ↓
Postprocess
 ↓
Overlay
 ↓
Display
```

ต้องวัดแยก:

```text
RTSP/decode CPU
Preprocess CPU
H2D transfer
GPU inference
D2H transfer
Postprocess CPU
Overlay CPU
Display CPU
```

มีความเป็นไปได้ว่า GPU ยังไม่เต็ม แต่ CPU หรือ decode เต็มก่อน

---

# 23. State Reset / Lifecycle

CameraEngine ต้องมี explicit lifecycle:

```python
reset()
update_config()
on_disconnect()
on_reconnect()
```

กรณี:

- RTSP disconnect
- reconnect
- camera restart
- config change
- video source change
- stream seek/test replay

ต้องกำหนดว่าจะ reset state ใดบ้าง

ตัวอย่างเมื่อ reconnect:

```text
Tracker state → reset
Temporal state → reset
Fall state → reset
Alert state → policy-dependent
```

ไม่ควรนำ stale state จากก่อน disconnect กลับมาใช้โดยไม่ตั้งใจ

---

# 24. Runtime Config Changes

`roles`, `ppe_items`, `zones` อาจเปลี่ยนขณะระบบกำลังทำงาน

ต้องออกแบบ:

```python
camera_engine.update_config(new_config)
```

ไม่ควรบังคับสร้าง engine ใหม่ทุกครั้ง

Architecture:

```text
Shared Models
    ↓
Immutable / shared

Camera Config
    ↓
Mutable

Camera State
    ↓
Resettable / reconfigurable
```

---

# 25. Display Must Be Decoupled

Display ไม่ควร block inference

ต้องสามารถ:

```text
Capture → Display
Capture → AI
```

ทำงานแยกกัน

AI result ล่าสุดอาจถูก overlay บน frame ใหม่

เช่น:

```text
Frame 100 → AI result
Frame 101 → reuse latest result
Frame 102 → reuse latest result
Frame 103 → new AI result
```

ต้องระวัง stale overlay และควรมี result timestamp/frame_id เพื่อกำหนด policy

---

# 26. Latency Is More Important Than Raw FPS

สำหรับ CCTV safety monitoring ไม่ควรวัดแค่ FPS

ต้องวัด:

```text
Frame timestamp
        ↓
Inference completed
        ↓
CameraEngine processed
        ↓
Display/Event emitted
```

แล้วคำนวณ:

```text
End-to-End Latency
```

ตัวอย่าง:

```text
Inference FPS = 10
Latency = 2.5 seconds
```

ถือว่าไม่ดี แม้ FPS ดูสูง

---

# 27. Recommended Metrics

ต้องเก็บอย่างน้อย:

### Per Camera

```text
capture_fps
inference_fps
display_fps
frame_age_ms
inference_latency_ms
end_to_end_latency_ms
dropped_frames
dropped_inference_tasks
```

### GPU

```text
gpu_utilization
vram_used
vram_total
gpu_temperature
inference_time
batch_size
```

### CPU

```text
cpu_usage
decode_time
preprocess_time
postprocess_time
```

---

# 28. Benchmark Matrix

ต้อง benchmark อย่างน้อย:

| Cameras | Person | PPE | Pose | Fall | Batch | VRAM | GPU | CPU | Latency |
|---:|---:|---:|---|---|---:|---:|---:|---:|---:|
| 1 | 10 | 10 | ON | ON | 1 | ? | ? | ? | ? |
| 2 | 10 | 10 | ON | ON | 2 | ? | ? | ? | ? |
| 4 | 10 | 5 | OFF | ON | 4 | ? | ? | ? | ? |
| 8 | 5 | 3 | OFF | OFF | 8 | ? | ? | ? | ? |
| 12 | 3 | 2 | OFF | OFF | 8 | ? | ? | ? | ? |

ค่าจริงต้องวัดบน hardware target

---

# 29. Verification Plan

## Phase 1 — Correctness / Parity

รัน:

```text
Original PPEEngine
vs
ModelHub + CameraEngine
```

บน video เดียวกัน

ตรวจ:

- detections
- boxes
- track IDs
- PPE association
- temporal confirmation
- zones
- roles
- alerts
- cooldown
- de-dup

ต้องหาความแตกต่างและอธิบายได้

---

## Phase 2 — Scheduler

ทดสอบ:

```text
1 camera
2 cameras
4 cameras
8 cameras
```

ตรวจ:

- batch formation
- result routing
- no camera starvation
- no stale queue
- frame freshness
- graceful FPS degradation

---

## Phase 3 — Scaling

วัด:

```text
VRAM
GPU
CPU
FPS
Latency
Frame age
Dropped inference
```

เป้าหมาย:

> VRAM ไม่โตแบบ N × model memory

---

## Phase 4 — Per-Camera Configuration

ตัวอย่าง:

```text
Camera A → Helmet
Camera B → Fall
Camera C → Zone
Camera D → Helmet + Vest
```

ต้องยืนยันว่าแต่ละกล้องใช้ requirement ของตัวเองจริง

---

## Phase 5 — Regression

ตรวจ:

```text
3-of-5 confirmation
cooldown
de-dup
zone enter/exit
PPE association
track lifecycle
fall confirmation
```

---

# 30. Suggested File Structure

เริ่มต้นประมาณนี้:

```text
backend/
├── utils/
│   ├── model_hub.py
│   ├── camera_engine.py
│   ├── tracker.py
│   ├── ppe_association.py
│   ├── temporal.py
│   ├── zone_geometry.py
│   └── detect_track.py
│
pipeline/
├── manager.py
├── scheduler.py
└── capture_manager.py
```

ชื่อไฟล์สามารถปรับตาม project convention

---

# 31. Important Non-Goals

ยังไม่ควรรีบทำ:

- TensorRT
- CUDA kernel optimization
- Cloud inference
- distributed inference
- multiprocessing complexity
- aggressive GPU optimization

ก่อนพิสูจน์ architecture และ correctness

ลำดับที่แนะนำ:

```text
Correctness
    ↓
Data Flow
    ↓
State Isolation
    ↓
Scheduling
    ↓
Batching
    ↓
Benchmark
    ↓
FP16
    ↓
TensorRT
    ↓
Adaptive Scaling
    ↓
Cloud
```

---

# 32. Recommended Architecture Priorities

ให้ความสำคัญตามลำดับ:

## P0 — Critical

1. Latest-frame / backpressure architecture
2. Timestamp + frame_id
3. Result routing
4. Per-camera tracker/state isolation
5. Model sharing

## P1 — High

6. Scheduler
7. Per-model FPS budget
8. Batch inference
9. Camera requirements
10. Fall throttling

## P2 — Optimization

11. Adaptive degradation
12. Resource Manager
13. Dynamic batch size
14. FP16
15. TensorRT

---

# 33. Questions Claude Code Must Answer Before Editing

ก่อนแก้ code ให้ตรวจ repository จริงและตอบคำถามเหล่านี้:

### Architecture

1. ปัจจุบัน `PPEEngine` โหลด model ที่จุดใดบ้าง?
2. มี model instance ซ้ำตรงไหน?
3. `Pipeline` มี thread/process/queue อะไร?
4. `MultiCameraManager` เป็น owner ของ lifecycle อะไร?
5. display pipeline ผูกกับ inference มากแค่ไหน?

### Data Flow

6. Frame ปัจจุบันถูกส่งผ่าน object ไหน?
7. มี frame queue หรือ latest-frame buffer แล้วหรือไม่?
8. มี frame_id/timestamp หรือไม่?
9. มีจุดไหนที่ copy frame ซ้ำโดยไม่จำเป็น?
10. OpenCV decode ใช้ CPU มากแค่ไหน?

### Inference

11. Model ใดเป็น bottleneck?
12. Person/PPE/Pose/Fall สามารถ schedule แยกกันได้หรือไม่?
13. Ultralytics batch API รองรับรูปแบบที่ใช้อยู่หรือไม่?
14. `_to_norm_dicts` สามารถ reuse ได้ตรงไหน?
15. ModelHub ควรเป็น singleton จริงหรือควรเป็น dependency-injected shared instance?

### Tracking

16. Standalone ByteTracker behavior ต่างจาก native tracker อย่างไร?
17. ต้อง reset tracker เมื่อใด?
18. ID continuity มีผลต่อ temporal state อย่างไร?

### State

19. state ใดเป็น per-camera?
20. state ใดเป็น per-track?
21. state ใดสามารถ reset ได้?
22. config change ต้อง reset state ใด?

### Scheduler

23. scheduler จะเป็น thread เดียวหรือ async worker?
24. GPU inference ถูกเรียกจาก thread ไหน?
25. batch formation ใช้ timeout หรือ target batch size?
26. ถ้ามีกล้อง 1 ตัว frame สด แต่กล้องอื่น stale จะ prioritize อย่างไร?
27. จะป้องกัน camera starvation อย่างไร?
28. จะ handle model-specific queues อย่างไร?

### Performance

29. CPU decode เป็น bottleneck หรือไม่?
30. CPU→GPU copy เป็น bottleneck หรือไม่?
31. GPU inference เป็น bottleneck หรือไม่?
32. batch size ที่เหมาะสมคือเท่าไร?
33. RTX 3050 4GB รองรับ workload เท่าไร?

---

# 34. Required Design Before Coding

Claude Code ควรเสนอ design ก่อนแก้ไฟล์หลัก โดยต้องแสดง:

1. Class diagram
2. Thread ownership
3. Queue/buffer ownership
4. Data classes / contracts
5. Scheduler flow
6. ModelHub API
7. CameraEngine API
8. Lifecycle / reset behavior
9. Error/reconnect behavior
10. Performance instrumentation
11. Test strategy
12. Migration strategy

อย่า refactor ทั้ง repository ในครั้งเดียวโดยไม่มี migration checkpoints

---

# 35. Suggested Migration Plan

## Step 1

สร้าง:

```text
model_hub.py
camera_engine.py
```

แต่ยังไม่เปลี่ยน multi-camera scheduler

เป้าหมาย:

```text
1 Camera
ModelHub + CameraEngine
≈ Original PPEEngine
```

---

## Step 2

ทำ parity tests

```text
Original
vs
Refactored
```

ต้องผ่านก่อน

---

## Step 3

สร้าง:

```text
scheduler.py
```

เริ่มจาก:

```text
latest frame
→ batch
→ ModelHub
→ result router
```

ยังไม่ต้อง adaptive scheduling ซับซ้อน

---

## Step 4

เชื่อม:

```text
MultiCameraManager
→ Scheduler
→ CameraEngine[]
```

---

## Step 5

เพิ่ม:

- target FPS
- priority
- freshness
- model requirements
- graceful degradation

---

## Step 6

Benchmark:

```text
1 / 2 / 4 / 8 cameras
```

---

## Step 7

Optimize:

```text
model size
imgsz
batch size
FP16
TensorRT
```

---

# 36. Final Design Goal

Architecture ที่ต้องการไม่ใช่เพียง:

> "โหลดโมเดลครั้งเดียว"

แต่คือ:

> **Shared AI Compute + Independent Camera State + Fresh Frame Scheduling + Graceful Resource Scaling**

เป้าหมายสุดท้าย:

```text
             Shared GPU
                 │
        ┌────────┴────────┐
        │                 │
    Shared Models     Scheduler
        │                 │
        └────────┬────────┘
                 │
        ┌────────┼────────┐
        ▼        ▼        ▼
      Cam 0    Cam 1    Cam N
        │        │        │
     Tracker  Tracker  Tracker
     Temporal Temporal Temporal
     Zone     Zone     Zone
     Role     Role     Role
        │        │        │
        └────────┼────────┘
                 ▼
              Events
```

ระบบควรสามารถเพิ่มกล้องได้โดย:

```text
1 → 2 → 4 → 8 → N
```

โดย behavior เป็น:

```text
FPS/camera ค่อย ๆ ลด
       +
latency อยู่ใน budget
       +
VRAM ไม่โตตามจำนวนกล้อง
       +
ระบบไม่ crash
       +
กล้องแต่ละตัวไม่ปน state กัน
```

---

# 37. Most Important Principle

ถ้าต้องเลือกเพียง 5 เรื่องเพื่อป้องกัน architecture พังภายหลัง ให้ prioritize:

1. **Latest-frame / backpressure**
2. **Timestamp + frame_id**
3. **Scheduler + per-model FPS budget**
4. **Independent per-camera state**
5. **Explicit result routing**

ModelHub เป็นส่วนที่ค่อนข้างตรงไปตรงมาเมื่อเทียบกับ 5 ข้อนี้

ดังนั้นอย่าเริ่มจากการเขียน model loader ก่อน

ให้เริ่มจาก:

```text
Data Contract
    ↓
Frame Ownership
    ↓
Scheduler Contract
    ↓
Result Routing
    ↓
CameraEngine State
    ↓
ModelHub
    ↓
Implementation
```

---

# 38. Instruction to Claude Code

ก่อนแก้ code:

1. Inspect repository จริงทั้งหมดที่เกี่ยวข้อง
2. ตรวจ architecture ปัจจุบัน
3. ห้าม assume ว่าไฟล์/ฟังก์ชันเป็นไปตามเอกสารนี้ทุกจุด
4. ระบุจุดที่ architecture ปัจจุบันขัดกับ proposal
5. เสนอ revised architecture
6. แสดง dependency graph
7. แสดง thread/queue ownership
8. ระบุ risk และ migration strategy
9. เริ่มแก้จาก smallest safe step
10. หลังแต่ละ phase ให้รัน test/parity ก่อนทำ phase ถัดไป

### ห้าม

- ลบ behavior เดิมโดยไม่มี parity test
- เปลี่ยน alert semantics โดยไม่ได้ตั้งใจ
- แชร์ tracker ระหว่างกล้อง
- ใช้ unbounded frame queue
- ผูก display FPS กับ inference FPS
- ให้ CameraEngine โหลด model เอง
- ให้ ModelHub ถือ camera-specific state
- ทำ inference task ที่ไม่จำเป็นต่อ role ของกล้อง
- optimize ด้วย TensorRT ก่อน architecture correctness

### ต้องมี

- explicit `camera_id`
- explicit `frame_id`
- explicit `timestamp`
- latest-frame semantics
- bounded workload
- per-camera state
- shared model
- batched inference
- graceful degradation
- metrics/instrumentation
- parity/regression tests

---

# 39. Expected Outcome

หลัง refactor ระยะแรก ต้องสามารถตอบได้ด้วยตัวเลขจริงว่า:

```text
1 camera:
VRAM = ?
FPS = ?
Latency = ?

2 cameras:
VRAM = ?
FPS/camera = ?
Latency = ?

4 cameras:
VRAM = ?
FPS/camera = ?
Latency = ?

8 cameras:
VRAM = ?
FPS/camera = ?
Latency = ?
```

และต้องตอบได้ว่า bottleneck อยู่ที่:

```text
Decode?
CPU?
Preprocess?
H2D?
GPU?
Postprocess?
Display?
Scheduler?
```

ไม่ใช่เพียงดูว่า "FPS ลดลง"

---

# 40. End Goal for ZENTRA

ZENTRA ควร evolve จาก:

```text
Camera
   ↓
Full AI Engine
```

เป็น:

```text
Many Cameras
     ↓
Shared Capture Layer
     ↓
Fresh Frame Scheduler
     ↓
Shared AI ModelHub
     ↓
Explicit Result Routing
     ↓
Independent Camera Engines
     ↓
Events / Alerts / Display
```

นี่คือ foundation ที่เหมาะกับ Edge AI และเปิดทางให้:

```text
RTX 3050
    ↓
larger edge GPU
    ↓
TensorRT
    ↓
Cloud ModelHub
```

โดยไม่ต้องเปลี่ยน business logic ของ CameraEngine ใหม่ทั้งหมด
