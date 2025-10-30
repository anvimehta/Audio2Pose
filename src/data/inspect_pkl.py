"""Load a 3D keypoints pickle, print keys and shapes, and plot the first frame."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np


def _as_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def _print_shapes(d: Dict[str, Any]) -> None:
    print("Keys:", list(d.keys()))
    for k, v in d.items():
        arr = None
        if isinstance(v, (list, tuple)) and len(v) > 0:
            try:
                arr = _as_numpy(v[0]) if hasattr(v[0], "shape") else None
            except Exception:
                arr = None
        if hasattr(v, "shape"):
            arr = _as_numpy(v)
        if arr is not None:
            print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}")
        else:
            print(f"  {k}: type={type(v)}")


def _find_pose_array(d: Dict[str, Any]) -> np.ndarray | None:
    candidate_keys = [
        "keypoints3d",
        "joints3d",
        "joints_3d",
        "poses_3d",
        "positions_3d",
        "pred_3d_joints",
    ]
    for k in candidate_keys:
        if k in d:
            arr = _as_numpy(d[k])
            if arr.ndim == 3 and arr.shape[-1] == 3:
                return arr.astype(np.float32)
    for v in d.values():
        if isinstance(v, dict):
            arr = _find_pose_array(v)
            if arr is not None:
                return arr
    return None


def _plot_first_frame_3d(pose_seq: np.ndarray) -> None:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    frame = pose_seq[0]
    J = frame.shape[0]
    connections = [
        (0, 1), (1, 2), (2, 3),
        (0, 4), (4, 5), (5, 6),
        (0, 7), (7, 8), (8, 9),
        (3, 10), (10, 11), (11, 12),
        (3, 13), (13, 14), (14, 15),
        (2, 16), (16, 17),
    ]
    connections = [(a, b) for a, b in connections if a < J and b < J]

    xs, ys, zs = frame[:, 0], frame[:, 1], frame[:, 2]
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xs, ys, zs, c="k", s=8)
    for a, b in connections:
        seg = np.stack([frame[a], frame[b]], axis=0)
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], c="tab:orange", linewidth=2)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=10, azim=-70)
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    plt.show()


def load_pkl(file_path: str | Path) -> dict:
    path = Path(file_path)
    with open(path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise TypeError(f"Expected dict at root of pkl, got {type(data)}")

    _print_shapes(data)

    pose = _find_pose_array(data)
    if pose is not None and pose.ndim == 3 and pose.shape[-1] == 3 and pose.shape[0] > 0:
        _plot_first_frame_3d(pose)
    else:
        print("No 3D joints array found for plotting.")

    return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect AIST++ 3D keypoints pkl")
    parser.add_argument("file", type=str, help="Path to .pkl file")
    args = parser.parse_args()
    load_pkl(args.file)


