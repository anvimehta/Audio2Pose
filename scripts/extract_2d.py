#!/usr/bin/env python
"""Extract 2D keypoints from PKL files to NPY format."""
from __future__ import annotations

import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).resolve().parents[1]
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from data.extract import extract_2d_keypoints


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Extract 2D keypoints from PKL files to NPY")
    parser.add_argument(
        "--pkl_dir",
        type=Path,
        default=Path("data/hiphop_la/2d_pkl"),
        help="Directory with 2D PKL files (default: data/hiphop_la/2d_pkl)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/hiphop_la/2d_npy"),
        help="Output directory for NPY files (default: data/hiphop_la/2d_npy)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Extract specific camera (0-8). If None, saves all cameras as (C, T, J, 3)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    pkl_dir = args.pkl_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    pkl_files = sorted(pkl_dir.glob("*.pkl"))
    if not pkl_files:
        print(f"No PKL files found in {pkl_dir}")
        return

    print(f"Found {len(pkl_files)} PKL files")
    print(f"Extracting to {output_dir}")
    if args.camera is not None:
        print(f"Extracting camera {args.camera} only")
    else:
        print("Extracting all cameras (shape: 9, T, J, 3)")

    success_count = 0
    skip_count = 0
    error_count = 0

    for pkl_file in pkl_files:
        try:
            if args.camera is not None:
                suffix = f"_c{args.camera+1:02d}"
            else:
                suffix = ""
            output_file = output_dir / f"{pkl_file.stem}{suffix}.npy"

            if output_file.exists() and not args.overwrite:
                skip_count += 1
                continue

            extract_2d_keypoints(pkl_file, output_dir, camera_idx=args.camera)
            success_count += 1
            if success_count % 10 == 0:
                print(f"Processed {success_count}/{len(pkl_files)} files...")

        except Exception as e:
            error_count += 1
            print(f"Error processing {pkl_file.name}: {e}")

    print(f"\nDone!")
    print(f"  Success: {success_count}")
    print(f"  Skipped: {skip_count}")
    print(f"  Errors: {error_count}")
    print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    main()

