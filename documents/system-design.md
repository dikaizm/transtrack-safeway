# Transtrack – Road Safety Hazard Detection
## System Design

---

## 1. Project Scope

Detect road safety hazards on **unpaved mining roads** from MDVR dashcam footage.
The system covers not only road surface damage but broader road disruptions that affect
mining vehicle safety — hence the scope is **Road Safety Hazard Detection**, not merely
Road Damage Detection.

---

## 2. Label Set

### 2.1 Final Classes (4)

| Label | Definition | Detection Zone |
|---|---|---|
| `road_depression` | Structural cavity or sunken area in road surface. Detectable by visible cavity edges and shadow, even when dry. | Road area only |
| `mud_patch` | Localized wet/muddy flat surface area visually distinct from surrounding road. Only label when contrast with surroundings is clear. | Road area only |
| `soil_mound` | Elevated pile of soil or debris above the road surface — physical obstruction. | Road area only |
| `traffic_sign` | Road safety signage (speed limit, hazard warning, stop, etc.) mounted on road-side. | Full frame |

### 2.2 Segmentation Classes (2)

| Label | Definition | Annotation Rule |
|---|---|---|
| `drive_area` | The navigable road surface where the haul truck is expected to travel. Includes the full width of the graded dirt/gravel road surface, road shoulders that are physically passable, and areas between ruts. | Label the continuous drivable surface from edge to edge. Include the road body even if wet, muddy, or rutted — the surface condition is handled by the detection model, not the segmentation. Dilate mask ~20px inward from road edges to absorb boundary ambiguity. |
| `off_road` | Everything outside the navigable road surface: road-side embankments, drainage ditches, vegetation, rock walls, sky, and any other non-drivable terrain. | Label any region not covered by `drive_area`. Sky is left unlabeled (background class) — do not mask it as `off_road`. |

**Key rules:**
- One road per frame — `drive_area` is a single connected region in most frames.
- Do **not** label `off_road` separately; it is the implicit complement of `drive_area`. Annotate `drive_area` only.
- Sky and distant background → leave as unlabeled background, not `off_road`.

---

### 2.4 Annotation Rules (Detection)

- **`road_depression`**: Label based on visible cavity edges and shadow. If the depression is filled with mud/water but the rim is still visible → label as `road_depression`. If no structural edge is visible → label as `mud_patch`.
- **`mud_patch`**: Only label when the wet/muddy area is locally distinct from surrounding road surface. If the entire road is uniformly wet → do **not** label anything.
- **`soil_mound`**: Label only within the road area, not road-side terrain.
- **`traffic_sign`**: Label the sign board itself, not the pole.
- **Ambiguous cases**: Skip the frame. A skipped ambiguous frame is better than a noisy label.

### 2.5 Rationale for Design Decisions

| Decision | Reason |
|---|---|
| `pothole` replaced by `road_depression` | "Pothole" implies paved asphalt context. "Depression" is accurate for unpaved dirt road cavities. |
| `mud_patch` kept separate from `road_depression` | Different cause (environmental vs structural), different risk (traction loss vs vehicle damage), different remediation action. |
| `traffic_sign` included | Mining safety compliance — speed zones, hazard warnings. Signs are detected full-frame, not constrained to road area. |
| `pothole` not merged with `mud_patch` | Distinct safety implications and remediation actions justify separation despite visual similarity in wet conditions. |

---

## 3. Dataset

### 3.1 Video Characteristics

| Property | Value |
|---|---|
| Resolution | 704×576 (standard), 1280×720 (1 special file) |
| Codec | H.264 (majority), HEVC (1 file) |
| FPS | 20fps (majority), 25, 40, 50fps |
| Avg duration | ~12.4 seconds per clip |
| Max duration | 41 seconds |
| Avg file size | ~1.5 MB |
| Max file size | 5.7 MB |

### 3.2 Conditions Coverage

| Condition | Clips | Status |
|---|---|---|
| Dry daytime | ~85 | Sufficient — label and train |
| Wet daytime | ~8 | Need more — collect or augment |
| Night | ~8 | Preprocess with CLAHE before labeling |

### 3.3 Training Augmentation

Apply during training (not at inference) to make the model robust to visual variation:

```yaml
hsv_h: 0.015      # hue shift — handles lighting variation
hsv_s: 0.7        # saturation — handles wet vs dry surface
hsv_v: 0.4        # brightness — handles overcast vs sunny
degrees: 0.0      # no rotation — camera is fixed forward-facing
translate: 0.1
scale: 0.5        # scale variation — near vs far objects
fliplr: 0.5       # horizontal flip
mosaic: 1.0       # combines 4 frames — YOLOv8 default
blur: 0.01        # motion blur from vehicle vibration
```

---

## 4. Pipeline Architecture

```
Client
  │
  POST /detect  (mp4, max 50MB, max 120s)
  │
  ▼
FastAPI (API Layer)
  ├── ffprobe validation (duration, codec, size) — reject before queuing
  └── store video to temp storage → enqueue task → return task_id
  ← 202 { "task_id": "abc123", "frames_to_process": 37 }

  GET /detect/{task_id}          → { "status": "pending|processing|done|failed" }
  GET /detect/{task_id}/result   → detection results (when status = done)

  ▼
Celery Worker (Redis broker)
  │
  ├─ 1. Frame Extraction
  │      ffmpeg @ 3fps (time-based, not frame-based)
  │      avg: ~37 frames/clip, max: ~123 frames/clip
  │
  ├─ 2. Condition Detection (per frame)
  │      avg brightness < 60        → night
  │      road area pixel variance   → uniform wet (suppress mud_patch)
  │
  ├─ 3. Preprocessing (per frame)
  │      night frame → CLAHE LAB (clipLimit=4.0, tileGridSize=8×8)
  │      dusty/hazy  → CLAHE LAB (clipLimit=2.0)
  │      normal day  → no preprocessing
  │
  ├─ 4. Road Segmentation  [YOLOv8n-seg — full frame]
  │      2 classes: drive_area / off_road
  │      → drive_area mask used as region-of-interest filter for hazard detection
  │      → night fallback: if mask confidence low → use center-frame polygon
  │
  ├─ 5. Safety Hazard Detection  [YOLOv8m — 4 classes]
  │      road_depression:  inside drive_area mask only
  │      mud_patch:        inside drive_area mask only
  │                        suppressed if road area pixel variance < threshold
  │      soil_mound:       inside drive_area mask only
  │      traffic_sign:     full frame — not masked, signs are on road-side
  │
  ├─ 6. Temporal Smoothing
  │      road_depression, mud_patch, soil_mound:
  │        must appear in ≥ 2 of 3 consecutive frames → reduces flicker noise
  │        (also handles haul-truck noise — trucks move fast enough to drop out)
  │      traffic_sign:
  │        ≥ 1 frame sufficient — static object, no temporal requirement
  │
  └─ 7. Persist & Cleanup
         save results to PostgreSQL
         mark task as done
         delete temp video file
```

---

## 5. Models

| Stage | Model | Input | Classes | Why |
|---|---|---|---|---|
| Road segmentation | YOLOv8n-seg | Full frame | `drive_area`, `off_road` | 2-class task — nano is sufficient; one road per frame |
| Safety hazard detection | YOLOv8m | Full frame | `road_depression`, `mud_patch`, `soil_mound`, `traffic_sign` | 4 classes, medium balances speed and accuracy |

### Notes
- All models use the same Ultralytics/YOLO toolchain — consistent training, inference, and export pipeline.
- Vehicle detection model removed — `drive_area` mask + temporal smoothing (≥2/3 frames) is sufficient to suppress haul-truck false positives.
- Revisit RF-DETR for the detection stage only if YOLOv8m mAP is insufficient after proper training.
- YOLOv8 is AGPL-3.0 licensed. If commercial deployment exposes this as a network service, confirm licensing with legal or switch to Apache-2.0 alternatives.

---

## 6. Preprocessing Detail

```python
import cv2
import numpy as np

NIGHT_BRIGHTNESS_THRESHOLD = 60
DUSTY_VARIANCE_THRESHOLD = 30
MUD_PATCH_VARIANCE_THRESHOLD = 15  # suppress mud_patch if road is uniformly wet

def detect_condition(frame: np.ndarray) -> str:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = gray.mean()
    variance = gray.std()
    if brightness < NIGHT_BRIGHTNESS_THRESHOLD:
        return "night"
    if variance < DUSTY_VARIANCE_THRESHOLD:
        return "dusty"
    return "normal"

def apply_clahe(frame: np.ndarray, clip: float) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

def preprocess(frame: np.ndarray) -> np.ndarray:
    condition = detect_condition(frame)
    if condition == "night":
        return apply_clahe(frame, clip=4.0)
    if condition == "dusty":
        return apply_clahe(frame, clip=2.0)
    return frame  # normal day — no preprocessing
```

**Important:** The same preprocessing applied at inference must be applied identically during training.

---

## 7. API Specification

### Endpoints

#### `POST /detect`
Submit a video for analysis.

- **Content-Type:** `multipart/form-data`
- **Body:** `video` (mp4 file)
- **Validation (fail-fast before queuing):**
  - File type: mp4, H.264 or HEVC codec
  - File size ≤ 50MB
  - Duration ≤ 120 seconds (probed via ffprobe)
- **Response:** `202 Accepted`

```json
{
  "task_id": "abc123",
  "frames_to_process": 37
}
```

---

#### `GET /detect/{task_id}`
Poll task status.

```json
{
  "task_id": "abc123",
  "status": "pending | processing | done | failed"
}
```

---

#### `GET /detect/{task_id}/result`
Retrieve results. Only accessible when `status = done`.

```json
{
  "task_id": "abc123",
  "status": "done",
  "conditions": "night | day | dusty",
  "frames_analyzed": 37,
  "detections": [
    {
      "frame_index": 8,
      "frame_sec": 2.67,
      "class": "traffic_sign",
      "confidence": 0.92,
      "bbox": { "x": 45, "y": 210, "w": 40, "h": 55 },
      "zone": "full_frame"
    },
    {
      "frame_index": 14,
      "frame_sec": 4.67,
      "class": "mud_patch",
      "confidence": 0.84,
      "bbox": { "x": 280, "y": 380, "w": 120, "h": 90 },
      "zone": "road_area"
    },
    {
      "frame_index": 21,
      "frame_sec": 7.0,
      "class": "road_depression",
      "confidence": 0.81,
      "bbox": { "x": 310, "y": 300, "w": 95, "h": 75 },
      "zone": "road_area"
    },
    {
      "frame_index": 25,
      "frame_sec": 8.33,
      "class": "soil_mound",
      "confidence": 0.89,
      "bbox": { "x": 200, "y": 320, "w": 110, "h": 85 },
      "zone": "road_area"
    }
  ]
}
```

**Field notes:**
- `frame_sec` — allows downstream consumer to seek to the exact moment in the video
- `zone` — `road_area` or `full_frame`, indicates which detection context was applied
- `conditions` — lets downstream API weight or flag detections accordingly

---

## 8. Infrastructure Stack

| Component | Technology |
|---|---|
| API layer | FastAPI |
| Task queue | Celery + Redis |
| Frame extraction | FFmpeg |
| Video validation | ffprobe |
| ML inference | Ultralytics (YOLOv8) |
| Result storage | PostgreSQL |
| Temp video storage | Local disk or MinIO/S3 |

### Operational Constraints

| Parameter | Value | Reason |
|---|---|---|
| Max file size | 50MB | 10× the largest observed training clip |
| Max duration | 120s | 3× the longest observed clip — safe buffer |
| Frame sampling | 3fps (time-based) | Sufficient for mining road speeds; ~37 frames avg |
| Worker timeout | 5 minutes | Covers worst-case 123 frames × multi-stage inference |
| Task result TTL | 24 hours | Results purged from DB after this period |
| Temp video retention | Until task completes | Deleted immediately after worker finishes |

---

## 9. Evaluation Strategy

- Evaluate **per-condition** separately: day / wet / night — never aggregate only
- Evaluate **per-class** mAP: road_depression, mud_patch, soil_mound, traffic_sign
- Train/val/test split at **video level**, not frame level — prevents temporal leakage
- Include uniformly wet road frames as **negative examples** for mud_patch
- Target metric: **mAP@50** per class per condition

---

## 10. Build Order

```
1. Finalize and label dataset (4 classes, annotation rules applied)
2. Train road segmentation model (YOLOv8n-seg)
3. Train safety hazard detection model (YOLOv8m)
4. Validate detection quality on sample clips (day, wet, night separately)
5. Build async API (FastAPI + Celery + Redis)
6. Integrate full pipeline into worker
7. End-to-end integration test
8. Per-condition evaluation and threshold tuning
```
