"""
Frame extraction and video validation using ffprobe + ffmpeg.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path


class VideoValidationError(Exception):
    pass


def probe_video(video_path: str) -> dict:
    """
    Run ffprobe and return video stream metadata.

    Raises VideoValidationError for unsupported or corrupt files.
    """
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise VideoValidationError("Could not read video file — may be corrupt or unsupported.")

    data = json.loads(result.stdout)
    fmt = data.get("format", {})
    streams = data.get("streams", [])
    video_stream = next((s for s in streams if s["codec_type"] == "video"), None)

    if not video_stream:
        raise VideoValidationError("No video stream found in file.")

    codec = video_stream.get("codec_name", "")
    if codec not in ("h264", "hevc", "h265"):
        raise VideoValidationError(
            f"Unsupported codec '{codec}'. Accepted: h264, hevc."
        )

    duration = float(fmt.get("duration", 0))
    size_bytes = int(fmt.get("size", 0))
    fps_raw = video_stream.get("r_frame_rate", "0/1")
    num, den = map(int, fps_raw.split("/"))
    fps = num / den if den else 0

    return {
        "duration": duration,
        "size_bytes": size_bytes,
        "fps": fps,
        "width": video_stream.get("width", 0),
        "height": video_stream.get("height", 0),
        "codec": codec,
    }


def validate_video(
    video_path: str,
    max_size_mb: int = 50,
    max_duration_sec: int = 120,
) -> dict:
    """
    Validate video constraints. Returns probe metadata on success.
    Raises VideoValidationError on failure.
    """
    size_bytes = os.path.getsize(video_path)
    if size_bytes > max_size_mb * 1024 * 1024:
        raise VideoValidationError(
            f"File too large: {size_bytes / 1024 / 1024:.1f}MB. Max: {max_size_mb}MB."
        )

    meta = probe_video(video_path)

    if meta["duration"] > max_duration_sec:
        raise VideoValidationError(
            f"Video too long: {meta['duration']:.1f}s. Max: {max_duration_sec}s."
        )

    return meta


def extract_frames(
    video_path: str,
    output_dir: str,
    sample_fps: int = 3,
) -> list[str]:
    """
    Extract frames from video at sample_fps using ffmpeg.
    Uses time-based sampling (not frame stepping) — consistent across varying FPS sources.

    Returns:
        Sorted list of extracted frame file paths.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_pattern = os.path.join(output_dir, "frame_%05d.jpg")

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", f"fps={sample_fps}",
        "-q:v", "2",        # JPEG quality (2 = high quality, low compression)
        output_pattern,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Frame extraction failed: {result.stderr}")

    frames = sorted(Path(output_dir).glob("frame_*.jpg"))
    return [str(f) for f in frames]
