"""
Visualization entry point — generate annotated result videos from a trained model.

Runs model.predict() on test images per condition and stitches annotated frames
into an MP4 video, one per condition. Also supports raw video input.

  - Segmentation : colored mask overlay (drive_area / off_road) + label
  - Detection    : bounding boxes + class label + confidence score

Videos are saved to runs/visualize/{run_name}/ and logged to MLflow.

Usage:
    # From per-condition test image dirs (default)
    python visualize.py --config config/yolo_segmentation.yaml --model runs/train/weights/best.pt
    python visualize.py --config config/yolo_detection.yaml   --model best.pt --conditions day night

    # From a raw MP4 video
    python visualize.py --config config/yolo_detection.yaml --model best.pt --source-video clip.mp4

    # Tune output
    python visualize.py --config config/yolo_detection.yaml --model best.pt --fps 3 --n-samples 60
"""

import argparse
import sys
from pathlib import Path

import cv2
import mlflow
import numpy as np
import yaml

import pipelines  # noqa: F401
from utils.gdrive import (
    build_service,
    get_run_visuals_folder,
    upload_and_share,
)
from utils.mlflow_helper import setup_mlflow, tag_gdrive_link
from utils.preprocess import preprocess_frame

# Playback FPS for the output video. 3fps matches the extraction rate used
# in the backend pipeline, so the video duration feels natural.
DEFAULT_FPS = 3


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    base_path = path.parent / "base.yaml"
    if base_path.exists() and str(base_path) != str(path):
        with open(base_path) as f:
            base = yaml.safe_load(f)
        merged = {**base, **cfg}
        for key in base:
            if isinstance(base.get(key), dict) and isinstance(cfg.get(key), dict):
                merged[key] = {**base[key], **cfg[key]}
        return merged
    return cfg


def _make_writer(out_path: Path, width: int, height: int, fps: int) -> cv2.VideoWriter:
    """Create an MP4 VideoWriter. Falls back to mp4v if avc1 is unavailable."""
    for fourcc_str in ("avc1", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        if writer.isOpened():
            return writer
    raise RuntimeError(f"Could not open VideoWriter for {out_path}")


def annotate_images(
    model,
    image_paths: list[Path],
    task: str,
    imgsz: int,
    device,
    conf: float = 0.25,
) -> list[np.ndarray]:
    """Run predict on a list of image paths, return list of annotated BGR frames."""
    results = model.predict(
        source=[str(p) for p in image_paths],
        imgsz=imgsz,
        device=device,
        verbose=False,
        conf=conf,
    )
    frames = []
    for result in results:
        annotated = result.plot(
            labels=True,
            conf=True,
            masks=(task == "segment"),
            boxes=True,
            line_width=2,
            font_size=12,
        )
        frames.append(annotated)
    return frames


def annotate_video(
    model,
    video_path: Path,
    task: str,
    imgsz: int,
    device,
    conf: float = 0.25,
) -> tuple[list[np.ndarray], int, int, int]:
    """
    Run predict on every frame of an MP4 video.
    Returns (annotated_frames, width, height, original_fps).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    orig_fps = int(cap.get(cv2.CAP_PROP_FPS)) or DEFAULT_FPS
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    raw_frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        raw_frames.append(frame)
    cap.release()

    if not raw_frames:
        return [], width, height, orig_fps

    print(f"  Read {len(raw_frames)} frames from {video_path.name}")

    results = model.predict(
        source=raw_frames,
        imgsz=imgsz,
        device=device,
        verbose=False,
        conf=conf,
    )
    annotated = [
        r.plot(labels=True, conf=True, masks=(task == "segment"),
               boxes=True, line_width=2, font_size=12)
        for r in results
    ]
    return annotated, width, height, orig_fps


def write_video(frames: list[np.ndarray], out_path: Path, fps: int) -> None:
    if not frames:
        return
    h, w = frames[0].shape[:2]
    writer = _make_writer(out_path, w, h, fps)
    for frame in frames:
        writer.write(frame)
    writer.release()


def burn_condition_label(frame: np.ndarray, condition: str, task: str) -> np.ndarray:
    """Overlay a small condition/task label in the top-left corner."""
    label = f"{task.upper()} | {condition}"
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (len(label) * 11 + 10, 30), (0, 0, 0), -1)
    cv2.putText(out, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def build_video_from_images(
    model,
    image_paths: list[Path],
    task: str,
    imgsz: int,
    device,
    out_path: Path,
    fps: int,
    condition: str,
    conf: float = 0.25,
) -> None:
    frames = annotate_images(model, image_paths, task, imgsz, device, conf)
    frames = [burn_condition_label(f, condition, task) for f in frames]
    write_video(frames, out_path, fps)


def collect_images(img_dir: Path, n_samples: int) -> list[Path]:
    all_imgs = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    if not all_imgs:
        return []
    # Evenly spaced sample so video spans the whole set
    step = max(1, len(all_imgs) // n_samples)
    return all_imgs[::step][:n_samples]


def run_visualization(
    config: dict,
    model_path: Path,
    conditions: list[str],
    n_samples: int,
    source_video: str | None,
    run_name: str,
    out_dir: Path,
    fps: int,
    conf: float,
) -> None:
    from ultralytics import YOLO

    task    = config.get("model", {}).get("task", "detect")
    test_cfg = config.get("data", {}).get("test", {})
    imgsz   = config.get("train", {}).get("imgsz", 640)
    device  = config.get("train", {}).get("device", 0)

    setup_mlflow(
        config["mlflow"]["tracking_uri"],
        config["mlflow"]["experiment_name"],
    )

    tags = {
        "model_type": "yolo",
        "task": task,
        "model_path": str(model_path),
        "type": "visualization",
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))

    with mlflow.start_run(run_name=run_name, tags=tags):
        mlflow.log_param("model_path", str(model_path))
        mlflow.log_param("n_samples", n_samples)
        mlflow.log_param("fps", fps)
        mlflow.log_param("conf_threshold", conf)
        mlflow.log_param("conditions", ",".join(conditions))

        # Build GDrive service once — reused for all uploads in this run
        try:
            svc = build_service()
            gdrive_folder_id = get_run_visuals_folder(svc, run_name)
            gdrive_ok = True
        except Exception as e:
            print(f"GDrive unavailable (non-fatal): {e}")
            svc = None
            gdrive_ok = False

        active_run_id = mlflow.active_run().info.run_id

        def _upload_video(out_path: Path, tag_key: str) -> None:
            mlflow.log_artifact(str(out_path), artifact_path="visuals")
            if gdrive_ok:
                try:
                    link = upload_and_share(svc, out_path, gdrive_folder_id)
                    tag_gdrive_link(active_run_id, tag_key, link)
                    print(f"  GDrive: {link}")
                except Exception as e:
                    print(f"  GDrive upload failed (non-fatal): {e}")

        if source_video:
            # Single raw video — run inference on every frame
            video_path = Path(source_video)
            print(f"\nRunning inference on video: {video_path.name}")
            frames, w, h, orig_fps = annotate_video(
                model, video_path, task, imgsz, device, conf
            )
            frames = [burn_condition_label(f, "custom", task) for f in frames]
            out_path = out_dir / f"result_{video_path.stem}.mp4"
            write_video(frames, out_path, fps or orig_fps)
            print(f"  Saved: {out_path} ({len(frames)} frames @ {fps}fps)")
            _upload_video(out_path, f"gdrive_vis_{task}_custom")

        else:
            # Per-condition test image dirs
            for condition in conditions:
                data_path = test_cfg.get(condition)
                if not data_path:
                    print(f"Skipping '{condition}' — not in config test paths")
                    continue

                img_dir = Path(data_path) / "images"
                if not img_dir.exists():
                    print(f"Skipping '{condition}' — images dir not found: {img_dir}")
                    continue

                images = collect_images(img_dir, n_samples)
                if not images:
                    print(f"Skipping '{condition}' — no images found in {img_dir}")
                    continue

                print(f"\n[{condition}] Annotating {len(images)} frames → video @ {fps}fps")

                out_path = out_dir / f"{task}_{condition}.mp4"
                build_video_from_images(
                    model, images, task, imgsz, device, out_path, fps, condition, conf
                )

                duration = len(images) / fps
                print(f"  Saved: {out_path.name}  ({len(images)} frames, ~{duration:.1f}s)")
                _upload_video(out_path, f"gdrive_vis_{task}_{condition}")
                print(f"  Logged to MLflow: visuals/{condition}/{out_path.name}")

        print(f"\nVideos saved to: {out_dir}")
        print(f"MLflow         : {config['mlflow']['tracking_uri']}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate annotated result videos from a trained YOLO model"
    )
    parser.add_argument("--config",   required=True, help="Config YAML (same one used for training)")
    parser.add_argument("--model",    required=True, help="Path to trained weights (best.pt)")
    parser.add_argument("--conditions", nargs="+", default=None,
                        choices=["all", "day", "wet", "night"],
                        help="Conditions to visualize (default: day wet night)")
    parser.add_argument("--source-video", default=None,
                        help="Raw MP4 input — run inference on every frame instead of test dirs")
    parser.add_argument("--n-samples", type=int, default=60,
                        help="Max frames to sample from test images per condition (default: 60)")
    parser.add_argument("--fps",      type=int, default=DEFAULT_FPS,
                        help=f"Output video FPS (default: {DEFAULT_FPS})")
    parser.add_argument("--conf",     type=float, default=0.25,
                        help="Confidence threshold for predictions (default: 0.25)")
    parser.add_argument("--run-name", default=None, help="MLflow run name")
    parser.add_argument("--out-dir",  default="runs/visualize",
                        help="Output directory for result videos (default: runs/visualize)")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {args.model}")
        sys.exit(1)

    config   = load_config(args.config)
    task     = config.get("model", {}).get("task", "detect")
    test_cfg = config.get("data", {}).get("test", {})

    conditions = args.conditions or [
        k for k in ["day", "wet", "night"] if Path(test_cfg.get(k, "")).exists()
    ] or ["all"]

    run_name = args.run_name or f"vis-{task}-{model_path.stem}"
    out_dir  = Path(args.out_dir) / run_name

    print(f"Task         : {task}")
    print(f"Model        : {model_path}")
    if args.source_video:
        print(f"Source video : {args.source_video}")
    else:
        print(f"Conditions   : {conditions}")
        print(f"Frames/cond  : {args.n_samples}")
    print(f"Output FPS   : {args.fps}")
    print(f"Conf thresh  : {args.conf}")
    print(f"Output dir   : {out_dir}")
    print()

    run_visualization(
        config=config,
        model_path=model_path,
        conditions=conditions,
        n_samples=args.n_samples,
        source_video=args.source_video,
        run_name=run_name,
        out_dir=out_dir,
        fps=args.fps,
        conf=args.conf,
    )


if __name__ == "__main__":
    main()
