#!/usr/bin/env python
"""Extract (T, J, 3) poses for all PKLs in a directory to .npy files."""
import argparse
import sys
from pathlib import Path

from tqdm import tqdm


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from data.extract import extract_poses

    parser = argparse.ArgumentParser(description="Batch extract 3D poses from PKLs")
    parser.add_argument("src_dir", type=str, help="Directory containing .pkl files")
    parser.add_argument("dst_dir", type=str, help="Directory to save .npy outputs")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .npy files")
    args = parser.parse_args()

    src = Path(args.src_dir)
    dst = Path(args.dst_dir)
    dst.mkdir(parents=True, exist_ok=True)

    pkls = sorted(src.glob("*.pkl"))
    if not pkls:
        print(f"No .pkl files found in {src}")
        return

    for p in tqdm(pkls, desc="Extracting"):
        out = dst / f"{p.stem}.npy"
        if out.exists() and not args.overwrite:
            continue
        try:
            extract_poses(str(p), str(dst))
        except Exception as e:
            print(f"Failed {p.name}: {e}")


if __name__ == "__main__":
    main()


