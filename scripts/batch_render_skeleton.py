#!/usr/bin/env python
"""Batch render skeleton videos for multiple files."""
from __future__ import annotations

import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).resolve().parents[1]
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import subprocess
from typing import List, Tuple, Optional


def find_matching_files(video_dir: Path, max_files: int = 10) -> List[tuple[Path, Path, Path]]:
    """
    Find matching video, pose, and audio files.
    
    Returns:
        List of tuples: (video_path, pose_path, audio_path)
    """
    video_files = sorted(list(video_dir.glob("*.mp4")))[:max_files]
    matches = []
    
    for video_path in video_files:
        # Extract base name (e.g., gLH_sBM_c01_d16_mLH0_ch01)
        video_stem = video_path.stem
        
        # Convert to pose file format (c01 -> cAll)
        parts = video_stem.split("_")
        if len(parts) >= 3 and parts[2].startswith("c"):
            parts[2] = "cAll"
            pose_stem = "_".join(parts)
        else:
            pose_stem = video_stem
        
        # Find pose file
        pose_path = Path(f"data/hiphop_la/3d_npy/{pose_stem}.npy")
        if not pose_path.exists():
            # Try with original name
            pose_path = Path(f"data/hiphop_la/3d_npy/{video_stem}.npy")
        
        # Find audio file - try exact match first
        audio_path = Path(f"data/hiphop_la/audio/{video_stem}.wav")
        if not audio_path.exists():
            # Try with cAll
            audio_path = Path(f"data/hiphop_la/audio/{pose_stem}.wav")
        
        if pose_path.exists():
            matches.append((video_path, pose_path, audio_path if audio_path.exists() else None))
        else:
            print(f"Warning: Could not find pose file for {video_path.name}")
    
    return matches


def main() -> None:
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch render skeleton videos")
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=Path("data/raw/aistpp_hiphop_la"),
        help="Directory containing video files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Output directory for rendered videos",
    )
    parser.add_argument(
        "--cameras",
        type=Path,
        default=Path("data/hiphop_la/cameras/setting6.json"),
        help="Path to camera parameters JSON",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=10,
        help="Maximum number of files to process (default: 10)",
    )
    parser.add_argument(
        "--joint-radius",
        type=int,
        default=6,
        help="Radius of joint spheres (default: 6)",
    )
    parser.add_argument(
        "--bone-thickness",
        type=int,
        default=4,
        help="Thickness of bone lines (default: 4)",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=5,
        help="Smoothing window size (default: 5)",
    )
    parser.add_argument(
        "--no-smooth",
        action="store_true",
        help="Disable pose smoothing",
    )
    parser.add_argument(
        "--show-labels",
        action="store_true",
        help="Show joint numbers on skeleton",
    )
    
    args = parser.parse_args()
    
    # Find matching files
    print(f"Searching for video files in {args.video_dir}...")
    matches = find_matching_files(args.video_dir, max_files=args.max_files)
    
    if not matches:
        print("No matching files found!")
        return
    
    print(f"\nFound {len(matches)} files to process:")
    for i, (video, pose, audio) in enumerate(matches, 1):
        audio_str = audio.name if audio else "N/A"
        print(f"  {i}. {video.name} -> {pose.name} (audio: {audio_str})")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each file
    print(f"\nProcessing {len(matches)} files...")
    print("=" * 60)
    
    for i, (video_path, pose_path, audio_path) in enumerate(matches, 1):
        print(f"\n[{i}/{len(matches)}] Processing: {video_path.name}")
        print("-" * 60)
        
        # Generate output filename
        output_name = video_path.stem + "_skeleton.mp4"
        output_path = args.output_dir / output_name
        
        # Build command
        cmd = [
            sys.executable,
            "scripts/render_skeleton_video.py",
            str(video_path),
            "--pose_3d", str(pose_path),
            "--cameras", str(args.cameras),
            "--output", str(output_path),
            "--joint-radius", str(args.joint_radius),
            "--bone-thickness", str(args.bone_thickness),
            "--smoothing-window", str(args.smoothing_window),
        ]
        
        if audio_path:
            cmd.extend(["--audio", str(audio_path)])
        
        if args.no_smooth:
            cmd.append("--no-smooth")
        
        if args.show_labels:
            cmd.append("--show-labels")
        
        # Run command
        try:
            result = subprocess.run(cmd, check=True, capture_output=False)
            print(f"✓ Completed: {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed: {video_path.name} (error code: {e.returncode})")
            continue
    
    print("\n" + "=" * 60)
    print(f"Batch processing complete! Processed {len(matches)} files.")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()

