#!/usr/bin/env python
"""Run the pickle inspector without module path issues."""
import argparse
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from data.inspect_pkl import load_pkl

    parser = argparse.ArgumentParser(description="Inspect AIST++ 3D keypoints pkl")
    parser.add_argument("file", type=str, help="Path to .pkl file")
    args = parser.parse_args()

    load_pkl(args.file)


if __name__ == "__main__":
    main()


