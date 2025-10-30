"""Extract (T, J, 3) 3D poses from an AIST++ PKL and save as .npy."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

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


def _find_pose_array(d: Any) -> np.ndarray | None:
    if isinstance(d, dict):
        for k in [
            "keypoints3d",
            "joints3d",
            "joints_3d",
            "poses_3d",
            "positions_3d",
            "pred_3d_joints",
        ]:
            if k in d:
                arr = _as_numpy(d[k])
                if arr.ndim == 3 and arr.shape[-1] == 3:
                    return arr.astype(np.float32)
        for v in d.values():
            out = _find_pose_array(v)
            if out is not None:
                return out
    if isinstance(d, (list, tuple)):
        for v in d:
            out = _find_pose_array(v)
            if out is not None:
                return out
    return None


def extract_poses(pkl_file: str | Path, save_dir: str | Path) -> Path:
    pkl_path = Path(pkl_file)
    dst_dir = Path(save_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    poses = _find_pose_array(data)
    if poses is None or poses.ndim != 3 or poses.shape[-1] != 3:
        raise ValueError("No (T, J, 3) pose array found in PKL")
    out_path = dst_dir / f"{pkl_path.stem}.npy"
    np.save(out_path, poses.astype(np.float32))
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract 3D poses to .npy")
    parser.add_argument("pkl_file", type=str)
    parser.add_argument("save_dir", type=str)
    args = parser.parse_args()
    out = extract_poses(args.pkl_file, args.save_dir)
    print(out)


