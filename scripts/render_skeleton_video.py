#!/usr/bin/env python
"""Render 3D skeleton visualization overlaid on video with audio sync."""
from __future__ import annotations

import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).resolve().parents[1]
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import argparse
import json
from typing import Dict, Optional

import cv2
import numpy as np

from utils.skeleton_viz import (
    load_camera_params,
    overlay_skeleton_on_video_frame,
)


def smooth_poses(poses: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Apply smoothing to pose sequence for fluid motion.
    
    Args:
        poses: Array of shape (T, J, 3)
        window_size: Size of smoothing window (must be odd)
    
    Returns:
        Smoothed poses
    """
    if window_size % 2 == 0:
        window_size += 1
    
    T, J, _ = poses.shape
    smoothed = poses.copy()
    half = window_size // 2
    
    # Apply 1D convolution along time axis for each joint
    for j in range(J):
        for dim in range(3):
            # Pad with edge values
            padded = np.pad(poses[:, j, dim], (half, half), mode="edge")
            # Simple moving average
            kernel = np.ones(window_size) / window_size
            smoothed[:, j, dim] = np.convolve(padded, kernel, mode="valid")
    
    return smoothed


def load_video_frames(video_path: Path) -> list[np.ndarray]:
    """Load all frames from video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Loading video: {video_path.name}")
    print(f"  FPS: {fps:.2f}, Frames: {frame_count}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    print(f"Loaded {len(frames)} frames")
    return frames, fps


def get_camera_name_from_video_path(video_path: Path) -> str:
    """Extract camera name from video filename (e.g., gLH_sBM_c01_... -> c01)."""
    # Video name format: gLH_sBM_c01_d16_mLH0_ch01.mp4
    # Extract camera name
    name = video_path.stem
    parts = name.split("_")
    
    # Find part that starts with 'c'
    for part in parts:
        if part.startswith("c") and len(part) >= 3 and part[1:].isdigit():
            return part  # e.g., "c01"
    
    # Default to c01 if not found
    return "c01"


def create_skeleton_video(
    video_path: Path,
    pose_3d_path: Path,
    camera_json_path: Path,
    audio_path: Optional[Path],
    output_path: Path,
    camera_name: Optional[str] = None,
    smooth: bool = True,
    smoothing_window: int = 5,
    joint_radius: int = 5,
    bone_thickness: int = 3,
    fps: Optional[float] = None,
    show_joint_labels: bool = False,
) -> None:
    """
    Create video with 3D skeleton overlay.
    
    Args:
        video_path: Path to input video
        pose_3d_path: Path to 3D pose NPY file (T, J, 3)
        camera_json_path: Path to camera parameters JSON
        audio_path: Path to audio file (optional)
        output_path: Path to output video
        camera_name: Camera name (e.g., "c01"). If None, extracted from video name
        smooth: Whether to smooth poses
        smoothing_window: Window size for smoothing
        joint_radius: Radius of joint spheres
        bone_thickness: Thickness of bone lines
        fps: Output FPS (if None, uses video FPS)
    """
    # Load data
    print("Loading data...")
    
    # Load 3D poses
    poses_3d = np.load(pose_3d_path).astype(np.float32)
    if poses_3d.ndim != 3 or poses_3d.shape[-1] != 3:
        raise ValueError(f"Expected poses with shape (T, J, 3), got {poses_3d.shape}")
    
    T, J, _ = poses_3d.shape
    print(f"Loaded 3D poses: shape {poses_3d.shape}")
    
    # Apply smoothing
    if smooth:
        print(f"Applying smoothing (window={smoothing_window})...")
        poses_3d = smooth_poses(poses_3d, window_size=smoothing_window)
    
    # Load camera parameters
    camera_params = load_camera_params(camera_json_path)
    print(f"Loaded camera parameters: {len(camera_params)} cameras")
    
    # Determine camera name
    if camera_name is None:
        camera_name = get_camera_name_from_video_path(video_path)
    
    if camera_name not in camera_params:
        raise ValueError(
            f"Camera {camera_name} not found in camera parameters. "
            f"Available: {list(camera_params.keys())}"
        )
    
    print(f"Using camera: {camera_name}")
    
    # Load video frames
    frames, video_fps = load_video_frames(video_path)
    
    if len(frames) != T:
        print(
            f"Warning: Video has {len(frames)} frames but poses have {T} frames. "
            f"Truncating to match."
        )
        min_len = min(len(frames), T)
        frames = frames[:min_len]
        poses_3d = poses_3d[:min_len]
        T = min_len
    
    # Get output dimensions
    H, W = frames[0].shape[:2]
    output_fps = fps if fps is not None else video_fps
    
    print(f"Output video: {W}x{H} @ {output_fps:.2f} fps")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, output_fps, (W, H))
    
    if not out.isOpened():
        raise RuntimeError(f"Could not create output video: {output_path}")
    
    # Process frames
    print("Processing frames...")
    for frame_idx in range(T):
        frame = frames[frame_idx]
        pose_3d = poses_3d[frame_idx]
        
        # Overlay skeleton
        frame_with_skeleton = overlay_skeleton_on_video_frame(
            frame,
            pose_3d,
            camera_params,
            camera_name=camera_name,
            joint_radius=joint_radius,
            bone_thickness=bone_thickness,
            show_joint_labels=show_joint_labels,
        )
        
        # Write frame
        out.write(frame_with_skeleton)
        
        if (frame_idx + 1) % 30 == 0:
            print(f"  Processed {frame_idx + 1}/{T} frames...")
    
    out.release()
    print(f"Video written to: {output_path}")
    
    # Add audio if provided
    if audio_path and audio_path.exists():
        print(f"Adding audio from: {audio_path}")
        
        # Use ffmpeg to combine video and audio
        import subprocess
        
        temp_video = output_path.with_suffix(".temp.mp4")
        output_path.rename(temp_video)
        
        cmd = [
            "ffmpeg",
            "-i",
            str(temp_video),
            "-i",
            str(audio_path),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            "-y",
            str(output_path),
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            temp_video.unlink()
            print(f"Final video with audio: {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"Warning: Could not add audio: {e}")
            print(f"  Using video without audio: {temp_video}")
            temp_video.rename(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render 3D skeleton visualization overlaid on video"
    )
    parser.add_argument("video_path", type=Path, help="Path to input video")
    parser.add_argument("--pose_3d", type=Path, help="Path to 3D pose NPY file")
    parser.add_argument(
        "--cameras",
        type=Path,
        default=Path("data/hiphop_la/cameras/setting6.json"),
        help="Path to camera parameters JSON",
    )
    parser.add_argument(
        "--audio",
        type=Path,
        help="Path to audio file (WAV/MP3)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output video",
    )
    parser.add_argument(
        "--camera",
        type=str,
        help="Camera name (e.g., c01). If not specified, extracted from video name",
    )
    parser.add_argument(
        "--no-smooth",
        action="store_true",
        help="Disable pose smoothing",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=5,
        help="Smoothing window size (default: 5)",
    )
    parser.add_argument(
        "--joint-radius",
        type=int,
        default=5,
        help="Radius of joint spheres (default: 5)",
    )
    parser.add_argument(
        "--bone-thickness",
        type=int,
        default=3,
        help="Thickness of bone lines (default: 3)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        help="Output FPS (if not specified, uses video FPS)",
    )
    parser.add_argument(
        "--show-labels",
        action="store_true",
        help="Show joint numbers on skeleton",
    )
    
    args = parser.parse_args()
    
    # Auto-detect pose file if not specified
    if args.pose_3d is None:
        # Extract file ID from video name
        video_stem = args.video_path.stem
        # Remove camera suffix (e.g., gLH_sBM_c01_d16_mLH0_ch01 -> gLH_sBM_cAll_d16_mLH0_ch01)
        parts = video_stem.split("_")
        # Replace camera-specific part with "cAll"
        if len(parts) >= 3 and parts[2].startswith("c"):
            parts[2] = "cAll"
        file_id = "_".join(parts)
        pose_3d_path = Path(f"data/hiphop_la/3d_npy/{file_id}.npy")
        
        if not pose_3d_path.exists():
            raise FileNotFoundError(
                f"Could not auto-detect pose file. Expected: {pose_3d_path}"
            )
        args.pose_3d = pose_3d_path
        print(f"Auto-detected pose file: {pose_3d_path}")
    
    # Auto-detect audio if not specified
    if args.audio is None:
        # Try to find matching audio file
        video_stem = args.video_path.stem
        
        # First try with exact video filename
        audio_path = Path(f"data/hiphop_la/audio/{video_stem}.wav")
        
        # If not found, try replacing camera part with "cAll"
        if not audio_path.exists():
            parts = video_stem.split("_")
            if len(parts) >= 3 and parts[2].startswith("c"):
                parts[2] = "cAll"
                file_id = "_".join(parts)
                audio_path = Path(f"data/hiphop_la/audio/{file_id}.wav")
        
        if audio_path.exists():
            args.audio = audio_path
            print(f"Auto-detected audio file: {audio_path}")
        else:
            print(f"Warning: Could not auto-detect audio file. Tried: {audio_path}")
    
    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Create video
    create_skeleton_video(
        video_path=args.video_path,
        pose_3d_path=args.pose_3d,
        camera_json_path=args.cameras,
        audio_path=args.audio,
        output_path=args.output,
        camera_name=args.camera,
        smooth=not args.no_smooth,
        smoothing_window=args.smoothing_window,
        joint_radius=args.joint_radius,
        bone_thickness=args.bone_thickness,
        fps=args.fps,
        show_joint_labels=args.show_labels,
    )


if __name__ == "__main__":
    main()

