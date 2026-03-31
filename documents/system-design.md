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

### 2.1 Detection Classes (4)

| Label | Definition | Detection Zone |
|---|---|---|
| `road_depression` | Structural cavity or sunken area in road surface. Detectable by visible cavity edges and shadow, even when dry. | Road area only |
| `mud_patch` | Localized wet/muddy flat surface area visually distinct from surrounding road. Only label when contrast with surroundings is clear. | Road area only |
| `soil_mound` | Elevated pile of soil or debris above the road surface — physical obstruction. | Road area only |
| `traffic_sign` | Road safety signage (speed limit, hazard warning, stop, etc.) mounted on road-side. | Full frame |

### 2.2 Segmentation Classes (2)

| Label | Definition | Annotation Rule |
|---|---|---|
| `drive_area` | The navigable road surface where the haul truck is expected to travel. Includes the full width of the graded dirt/gravel road surface, road shoulders that are physically passable, and areas between ruts. | Label the continuous drivable surface from edge to edge. Include the road body even if wet, muddy, or rutted — surface condition is handled by the detection model. Dilate mask ~20px inward from road edges to absorb boundary ambiguity. |
| `off_road` | Everything outside the navigable road surface: road-side embankments, drainage ditches, vegetation, rock walls, and any other non-drivable terrain. | Implicit complement of `drive_area`. Annotate `drive_area` only — `off_road` is inferred. Sky and distant background → leave as unlabeled background. |

**Key rules:**
- One road per frame — `drive_area` is a single connected region in most frames.
- Do **not** label `off_road` separately; annotate `drive_area` only.
- Sky and distant background → leave as unlabeled background, not `off_road`.

---

### 2.3 Annotation Rules (Detection)

- **`road_depression`**: Label based on visible cavity edges and shadow. If the depression is filled with mud/water but the rim is still visible → label as `road_depression`. If no structural edge is visible → label as `mud_patch`.
- **`mud_patch`**: Only label when the wet/muddy area is locally distinct from surrounding road surface. If the entire road is uniformly wet → do **not** label anything.
- **`soil_mound`**: Label only within the road area, not road-side terrain.
- **`traffic_sign`**: Label the sign board itself, not the pole.
- **Ambiguous cases**: Skip the frame. A skipped ambiguous frame is better than a noisy label.

### 2.4 Rationale for Design Decisions

| Decision | Reason |
|---|---|
| `pothole` replaced by `road_depression` | "Pothole" implies paved asphalt context. "Depression" is accurate for unpaved dirt road cavities. |
| `mud_patch` kept separate from `road_depression` | Different cause (environmental vs structural), different risk (traction loss vs vehicle damage), different remediation action. |
| `traffic_sign` included | Mining safety compliance — speed zones, hazard warnings. Signs are detected full-frame, not constrained to road area. |
| Vehicle detection removed | `drive_area` mask + temporal smoothing (≥2/3 frames) is sufficient to suppress haul-truck false positives. Trucks move fast enough to drop out of consecutive frames at 3fps. |

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

| Condition | Clips | Annotation FPS | Status |
|---|---|---|---|
| Dry daytime | ~85 | 1fps | Sufficient — label and train |
| Wet daytime | ~8 | 2fps | Need more — collect or augment |
| Night | ~8 | 2fps | Preprocess with CLAHE before labeling |

### 3.3 Roboflow Datasets

| Task | Project | Version | Format |
|---|---|---|---|
| Segmentation | `stelar/rdd-mining-road-seg` | v1 | yolov8 |
| Detection | `stelar/rdd-mining-road-det` | v4 | yolov8 |

Download:
```bash
python scripts/download_datasets.py --api-key <key>
```

### 3.4 Train/Val/Test Split

- Split at **video clip level**, not frame level — prevents temporal leakage
- **80 / 10 / 10** globally, stratified by condition
- Test set further split into `test/day/`, `test/wet/`, `test/night/` for per-condition evaluation

### 3.5 Training Augmentation

```yaml
# YOLO augmentation (applied during training only)
hsv_h: 0.015      # hue shift — handles lighting variation
hsv_s: 0.7        # saturation — handles wet vs dry surface
hsv_v: 0.4        # brightness — handles overcast vs sunny
degrees: 0.0      # no rotation — camera is fixed forward-facing
translate: 0.1
scale: 0.5        # scale variation — near vs far objects
fliplr: 0.5       # horizontal flip
mosaic: 1.0       # combines 4 frames — YOLOv8 default

# SegFormer augmentation
fliplr: 0.5
color_jitter: brightness=0.4, contrast=0.4
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
  ├─ 4. Road Segmentation  [YOLOv8n-seg · seg-v1 · mAP@50=0.995]
  │      2 classes: drive_area / off_road
  │      → drive_area mask used as region-of-interest filter for hazard detection
  │      → night fallback: if mask confidence low → use center-frame polygon
  │
  ├─ 5. Safety Hazard Detection  [YOLOv8m · det-v2 (v4 dataset) · mAP@50≈0.906]
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

### 5.1 Production Models

| Stage | Model | License | Classes | Notes |
|---|---|---|---|---|
| Road segmentation | YOLOv8n-seg | AGPL-3.0 | `drive_area`, `off_road` | Nano sufficient for 2-class semantic task; one road per frame |
| Safety hazard detection | YOLOv8m | AGPL-3.0 | `road_depression`, `mud_patch`, `soil_mound`, `traffic_sign` | Medium balances speed and accuracy for 4-class detection |

### 5.2 Comparative Models (Apache 2.0)

Trained on the same dataset for performance comparison. Use these if AGPL-3.0 licensing is a concern for commercial deployment.

| Stage | Model | License | Source | Notes |
|---|---|---|---|---|
| Road segmentation | SegFormer-B0 | Apache 2.0 | `nvidia/mit-b0` (HuggingFace) | Mix Transformer encoder; semantic segmentation; mIoU metric |
| Safety hazard detection | RT-DETR | Apache 2.0 | `PekingU/rtdetr_r50vd` (HuggingFace) | Real-time detection transformer; ResNet-50 backbone |

### 5.3 Licensing Note

- **AGPL-3.0 (YOLOv8)**: If deployed as a network service (SaaS), the source code must be made available. Confirm with legal before commercial deployment or switch to Apache-2.0 alternatives.
- **Apache 2.0 (SegFormer, RT-DETR)**: Permissive — safe for commercial deployment without source disclosure.

### 5.4 Training Results

#### Segmentation — YOLOv8n-seg (run: `seg-v1`, dataset: v1)

| Metric | Value |
|---|---|
| mAP@50 (Box) | **0.9950** |
| mAP@50-95 (Box) | 0.8954 |
| mAP@50 (Mask) | **0.9950** |
| mAP@50-95 (Mask) | 0.8138 |
| Precision | 0.9920 |
| Recall | 1.0000 |

MLflow run: [`seg-v1`](https://mlflow-geoai.stelarea.com/#/experiments/30/runs/f4b623dbdb754583be1236f6ab495cb2)

#### Detection — YOLOv8m (run: `det-v1`, dataset: v3)

| Metric | Value |
|---|---|
| mAP@50 | **0.9064** |
| mAP@50-95 | 0.5544 |
| Precision | 0.8739 |
| Recall | 0.8361 |

MLflow run: [`det-v1`](https://mlflow-geoai.stelarea.com/#/experiments/30/runs/ae9fbdd196314d24b92e0e25cdda070c)  
Weights (GDrive): https://drive.google.com/file/d/172tQOkwb-djOquHY0iHQ-HwL_cYH9T1I/view

> **Note:** `mud_patch` had zero instances in the val split — all wet-condition clips landed in train. Val mAP is effectively 3-class. Per-condition test evaluation on `test/wet/` is required to measure `mud_patch` performance.

#### Detection — YOLOv8m (run: `det-v2`, dataset: v4) — *training in progress*

Re-training with updated dataset v4 which adds `mud_patch` examples to the val split.

---

## 6. Preprocessing Detail

```python
NIGHT_BRIGHTNESS_THRESHOLD = 60
DUSTY_VARIANCE_THRESHOLD   = 30
MUD_PATCH_VARIANCE_THRESHOLD = 15   # suppress mud_patch if road is uniformly wet

def detect_condition(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if gray.mean() < NIGHT_BRIGHTNESS_THRESHOLD:
        return "night"
    if gray.std() < DUSTY_VARIANCE_THRESHOLD:
        return "dusty"
    return "day"

def preprocess_frame(frame):
    condition = detect_condition(frame)
    if condition == "night":
        return apply_clahe(frame, clip=4.0), condition
    if condition == "dusty":
        return apply_clahe(frame, clip=2.0), condition
    return frame, condition   # day — no preprocessing
```

**Important:** The same `preprocess_frame()` function is used at both training time and inference time. Any change here must be applied consistently to both.

---

## 7. API Specification

### Endpoints

#### `POST /detect`
Submit a video for analysis.

- **Content-Type:** `multipart/form-data`
- **Body:** `video` (mp4 file)
- **Validation:** file type, size ≤ 50MB, duration ≤ 120s (ffprobe)
- **Response:** `202 Accepted`

```json
{ "task_id": "abc123", "frames_to_process": 37 }
```

#### `GET /detect/{task_id}`
```json
{ "task_id": "abc123", "status": "pending | processing | done | failed" }
```

#### `GET /detect/{task_id}/result`
```json
{
  "task_id": "abc123",
  "status": "done",
  "conditions": "night",
  "frames_analyzed": 37,
  "detections": [
    {
      "frame_index": 8,
      "frame_sec": 2.67,
      "class": "traffic_sign",
      "confidence": 0.92,
      "bbox": { "x": 45, "y": 210, "w": 40, "h": 55 },
      "zone": "full_frame"
    }
  ]
}
```

**Field notes:**
- `frame_sec` — allows downstream consumer to seek to the exact moment in the video
- `zone` — `road_area` or `full_frame`
- `conditions` — lets downstream API weight or flag detections accordingly

---

## 8. Infrastructure Stack

| Component | Technology |
|---|---|
| API layer | FastAPI |
| Task queue | Celery + Redis |
| Frame extraction | FFmpeg |
| Video validation | ffprobe |
| ML inference | Ultralytics YOLOv8 (production) / HuggingFace Transformers (comparative) |
| Result storage | PostgreSQL |
| Temp video storage | Local disk |
| Experiment tracking | MLflow (`https://mlflow-geoai.stelarea.com/`) |
| Model & artifact storage | Google Drive (linked from MLflow tags) |

### Operational Constraints

| Parameter | Value | Reason |
|---|---|---|
| Max file size | 50MB | 10× the largest observed training clip |
| Max duration | 120s | 3× the longest observed clip |
| Frame sampling | 3fps (time-based) | Sufficient for mining road speeds; ~37 frames avg |
| Worker timeout | 5 minutes | Covers worst-case 123 frames × multi-stage inference |
| Task result TTL | 24 hours | Results purged from DB after this period |
| Temp video retention | Until task completes | Deleted immediately after worker finishes |

---

## 9. Model Development Pipeline

### 9.1 Training & Evaluation

All models are trained and evaluated through a unified pipeline with MLflow tracking.

```bash
# Full pipeline: download → train seg → eval seg → train det → eval det
bash run_all.sh --api-key <roboflow_key> --run-name v1

# Individual training
python train.py --config config/yolo_segmentation.yaml    --run-name yolo-seg-v1
python train.py --config config/yolo_detection.yaml       --run-name yolo-det-v1
python train.py --config config/segformer_segmentation.yaml --run-name segformer-v1
python train.py --config config/rtdetr_detection.yaml     --run-name rtdetr-v1

# Evaluation (per-condition: day / wet / night)
python evaluate.py --config config/yolo_detection.yaml --model runs/train/weights/best.pt

# Visual result videos (annotated MP4 per condition)
python visualize.py --config config/yolo_detection.yaml --model best.pt
python visualize.py --config config/yolo_detection.yaml --model best.pt --source-video clip.mp4
```

### 9.2 Artifact Storage

| Artifact | Location |
|---|---|
| Metrics, params, logs | MLflow (`transtrack-road-safety` experiment) |
| Model weights (`best.pt`) | Google Drive → linked as `gdrive_{task}_weights` MLflow tag |
| Result videos (annotated MP4) | Google Drive → linked as `gdrive_vis_{task}_{condition}` MLflow tag |

### 9.3 Evaluation Strategy

- Evaluate **per-condition** separately: day / wet / night — never aggregate only
- Evaluate **per-class**: `road_depression`, `mud_patch`, `soil_mound`, `traffic_sign`
- Split at **video clip level**, not frame level — prevents temporal leakage
- Include uniformly wet road frames as **negative examples** for `mud_patch`

| Model | Primary metric |
|---|---|
| YOLOv8n-seg, SegFormer-B0 | mIoU per class (drive_area, off_road) |
| YOLOv8m, RT-DETR | mAP@50 per class per condition |

---

## 10. Build Order

```
[✓] 1. Annotate dataset in Roboflow (detection v4 + segmentation v1)
[✓] 2. Download datasets: python scripts/download_datasets.py --api-key <key>
[ ] 3. Distribute test images into test/day/, test/wet/, test/night/
[✓] 4. Train segmentation: seg-v1 (YOLOv8n-seg, mAP@50=0.995) — DONE
[~] 5. Train detection: det-v2 (YOLOv8m, dataset v4) — IN PROGRESS
[ ] 6. Compare model results in MLflow (YOLOv8 vs Apache 2.0 alternatives)
[ ] 7. Select best models for production (segmentation + detection)
[ ] 8. Deploy weights to backend: update model_segmentation_weights + model_detection_weights in config.py
[ ] 9. End-to-end integration test with sample dashcam clips
[ ] 10. Per-condition threshold tuning
```
