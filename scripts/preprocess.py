#!/usr/bin/env python
"""Preprocess AIST++ LA Hip-Hop dataset: filter PKLs, extract poses, extract audio, extract beats/tempo, copy matching files."""
import argparse
import json
import pickle
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm


def _genre_from_filename(path: Path) -> str | None:
    name = path.stem
    parts = name.split("_")
    if not parts:
        return None
    token = parts[0]
    if not token.startswith("g") or len(token) < 3:
        return None
    code = token[1:3].upper()
    code_to_genre = {
        "BR": "break",
        "PO": "pop",
        "LO": "lock",
        "MH": "middle_hiphop",
        "LH": "la_hiphop",
        "HO": "house",
        "WA": "waack",
        "KR": "krump",
        "JS": "street_jazz",
        "JB": "ballet_jazz",
    }
    return code_to_genre.get(code)


def _find_pose_array(data: Any) -> Any:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from data.extract import _find_pose_array as _find
    return _find(data)


def extract_audio(video_path: Path, output_path: Path, sr: int = 16000) -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-i", str(video_path), "-vn", "-acodec", "pcm_s16le", "-ar", str(sr), "-ac", "1", "-f", "wav", "-y", str(output_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        return output_path.exists() and output_path.stat().st_size > 0
    except FileNotFoundError:
        return False


def infer_audio_from_name(pkl_path: Path, audio_root: Path | None) -> Path | None:
    name = pkl_path.stem
    if audio_root is None:
        return None
    p = audio_root / (name + ".wav")
    if p.exists():
        return p
    parts = name.split("_")
    if len(parts) >= 3 and parts[2] == "cAll":
        for cam in ["c01", "c02", "c03", "c04", "c05", "c06", "c07", "c08", "c09"]:
            parts[2] = cam
            candidate = "_".join(parts) + ".wav"
            p = audio_root / candidate
            if p.exists():
                return p
    return None


def extract_beats_tempo(audio_path: Path, output_path: Path, sr: int = 16000) -> bool:
    """Extract beat frames and tempo from audio using librosa."""
    try:
        import librosa
    except ImportError:
        print("Warning: librosa not available. Skipping beat/tempo extraction.")
        return False
    
    try:
        # Load audio
        y, sr_actual = librosa.load(str(audio_path), sr=sr, mono=True)
        
        # Extract tempo and beat frames
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        
        # Convert beat frames to timestamps (in seconds)
        beat_times = librosa.frames_to_time(beats, sr=sr)
        
        # Save as JSON with tempo (BPM) and beat timestamps
        data = {
            "tempo_bpm": float(tempo),
            "beat_times": beat_times.tolist(),  # Convert numpy array to list
            "beat_frames": beats.tolist(),
            "sample_rate": int(sr),
        }
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        
        return output_path.exists() and output_path.stat().st_size > 0
    except Exception as e:
        print(f"Failed to extract beats from {audio_path.name}: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess AIST++ LA Hip-Hop dataset")
    parser.add_argument("--pkl_dir", type=Path, help="Directory with all PKL files")
    parser.add_argument("--video_dir", type=Path, help="Directory with MP4 videos")
    parser.add_argument("--output_dir", type=Path, default=Path("data/hiphop_la"), help="Output directory")
    parser.add_argument("--skip_poses", action="store_true", help="Skip pose extraction")
    parser.add_argument("--skip_audio", action="store_true", help="Skip audio extraction")
    parser.add_argument("--skip_beats", action="store_true", help="Skip beat/tempo extraction")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    out_dir = args.output_dir
    pkl_dir = out_dir / "pkl"
    poses_dir = out_dir / "poses_npy"
    audio_dir = out_dir / "audio"
    beats_dir = out_dir / "beats"
    pkl_dir.mkdir(parents=True, exist_ok=True)
    poses_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    beats_dir.mkdir(parents=True, exist_ok=True)

    if args.pkl_dir:
        print("Filtering LA Hip-Hop PKLs...")
        pkls = sorted(args.pkl_dir.glob("*.pkl"))
        filtered = 0
        for pkl_path in tqdm(pkls, desc="Filtering"):
            genre = _genre_from_filename(pkl_path)
            if genre == "la_hiphop":
                dst = pkl_dir / pkl_path.name
                if not dst.exists() or args.overwrite:
                    shutil.copy2(pkl_path, dst)
                filtered += 1
        print(f"Filtered {filtered} LA Hip-Hop PKLs")

    if not args.skip_poses:
        print("\nExtracting poses...")
        repo_root = Path(__file__).resolve().parents[1]
        src_path = repo_root / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
        from data.extract import extract_poses

        pkls = sorted(pkl_dir.glob("*.pkl"))
        extracted = 0
        for p in tqdm(pkls, desc="Extracting poses"):
            out = poses_dir / f"{p.stem}.npy"
            if out.exists() and not args.overwrite:
                continue
            try:
                extract_poses(str(p), str(poses_dir))
                extracted += 1
            except Exception as e:
                print(f"Failed {p.name}: {e}")
        print(f"Extracted {extracted}/{len(pkls)} poses")

    if args.video_dir and not args.skip_audio:
        print("\nExtracting audio from videos...")
        video_audio_dir = args.video_dir / "audio"
        video_audio_dir.mkdir(exist_ok=True)
        videos = sorted(args.video_dir.glob("*.mp4"))
        extracted = 0
        for vid in tqdm(videos, desc="Extracting audio"):
            out = video_audio_dir / (vid.stem + ".wav")
            if out.exists() and not args.overwrite:
                continue
            if extract_audio(vid, out):
                extracted += 1
        print(f"Extracted {extracted}/{len(videos)} audio files")

        print("\nCopying matching audio files...")
        pkls = sorted(pkl_dir.glob("*.pkl"))
        copied = 0
        for pkl_path in tqdm(pkls, desc="Copying audio"):
            src_audio = infer_audio_from_name(pkl_path, video_audio_dir)
            if src_audio and src_audio.exists():
                dst_audio = audio_dir / src_audio.name
                if not dst_audio.exists() or args.overwrite:
                    shutil.copy2(src_audio, dst_audio)
                copied += 1
        print(f"Copied {copied}/{len(pkls)} audio files")

    # Extract beat/tempo annotations from audio files
    if not args.skip_beats:
        print("\nExtracting beat/tempo annotations...")
        audio_files = sorted(audio_dir.glob("*.wav"))
        extracted = 0
        for audio_path in tqdm(audio_files, desc="Extracting beats"):
            beats_output = beats_dir / f"{audio_path.stem}.json"
            if beats_output.exists() and not args.overwrite:
                continue
            if extract_beats_tempo(audio_path, beats_output):
                extracted += 1
        print(f"Extracted {extracted}/{len(audio_files)} beat/tempo annotations")

    print(f"\nPreprocessing complete! Output in {out_dir}")


if __name__ == "__main__":
    main()

