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
    parser.add_argument("--3d", action="store_true", dest="plot_3d", help="Show 3D plot")
    parser.add_argument(
        "--no-2d", action="store_false", dest="plot_2d", help="Don't show 2D stick figure plot"
    )
    parser.add_argument(
        "--frames", type=int, default=3, help="Number of frames to show in 2D plot (default: 3)"
    )
    parser.add_argument(
        "--view",
        type=str,
        default="front",
        choices=["front", "side", "top"],
        help="2D projection view: front, side, or top (default: front)",
    )
    parser.add_argument(
        "--animate", action="store_true", dest="plot_animated", help="Show animated 2D stick figure"
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Frames per second for animation (default: 30)"
    )
    parser.add_argument(
        "--save-animation",
        type=str,
        default=None,
        help="Save animation to file (e.g., output.gif or output.mp4)",
    )
    args = parser.parse_args()

    load_pkl(
        args.file,
        plot_3d=args.plot_3d,
        plot_2d=args.plot_2d,
        plot_animated=args.plot_animated,
        num_frames_2d=args.frames,
        view_2d=args.view,
        fps=args.fps,
        save_animation=args.save_animation,
    )


if __name__ == "__main__":
    main()


