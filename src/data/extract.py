"""Extract (T, J, 3) 3D poses or (C, T, J, 3) 2D keypoints from an AIST++ PKL and save as .npy."""
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


def extract_2d_keypoints(pkl_file: str | Path, save_dir: str | Path, camera_idx: int | None = None) -> Path:
    """
    Extract 2D keypoints from a 2D PKL file and save as .npy.
    
    Args:
        pkl_file: Path to 2D PKL file
        save_dir: Directory to save extracted .npy file
        camera_idx: If None, save all cameras (C, T, J, 3). If specified, save single camera (T, J, 3)
    
    Returns:
        Path to saved .npy file
    """
    pkl_path = Path(pkl_file)
    dst_dir = Path(save_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    
    if not isinstance(data, dict) or "keypoints2d" not in data:
        raise ValueError("No keypoints2d found in PKL file")
    
    keypoints2d = _as_numpy(data["keypoints2d"])
    
    if keypoints2d.ndim != 4 or keypoints2d.shape[-1] != 3:
        raise ValueError(f"Expected keypoints2d with shape (C, T, J, 3), got {keypoints2d.shape}")
    
    # Extract single camera or all cameras
    if camera_idx is not None:
        if camera_idx < 0 or camera_idx >= keypoints2d.shape[0]:
            raise ValueError(f"Camera index {camera_idx} out of range [0, {keypoints2d.shape[0]})")
        keypoints = keypoints2d[camera_idx]  # Shape: (T, J, 3)
        suffix = f"_c{camera_idx+1:02d}"
    else:
        keypoints = keypoints2d  # Shape: (C, T, J, 3)
        suffix = ""
    
    out_path = dst_dir / f"{pkl_path.stem}{suffix}.npy"
    np.save(out_path, keypoints.astype(np.float32))
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract 3D poses or 2D keypoints to .npy")
    parser.add_argument("pkl_file", type=str, help="Path to PKL file")
    parser.add_argument("save_dir", type=str, help="Directory to save .npy file")
    parser.add_argument("--2d", action="store_true", dest="is_2d", help="Extract 2D keypoints instead of 3D poses")
    parser.add_argument("--camera", type=int, default=None, help="Camera index for 2D extraction (0-8), or None for all cameras")
    args = parser.parse_args()
    
    if args.is_2d:
        out = extract_2d_keypoints(args.pkl_file, args.save_dir, camera_idx=args.camera)
    else:
        out = extract_poses(args.pkl_file, args.save_dir)
    print(out)


