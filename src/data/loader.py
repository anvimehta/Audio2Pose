"""Load a pose sequence and matching audio for a given AIST++ file id and plot the first frame."""
from __future__ import annotations

import glob
from pathlib import Path
from typing import Tuple

import numpy as np


def _find_one(patterns: list[str]) -> Path:
    for p in patterns:
        matches = glob.glob(p)
        if matches:
            return Path(matches[0])
    raise FileNotFoundError(f"No file found for patterns: {patterns}")


def _load_audio(audio_path: Path, target_sr: int | None = None) -> Tuple[np.ndarray, int]:
    try:
        import soundfile as sf

        wav, sr = sf.read(str(audio_path))
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
    except Exception:
        import librosa

        wav, sr = librosa.load(str(audio_path), sr=None, mono=True)

    if target_sr is not None and sr != target_sr:
        try:
            import librosa

            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        except Exception as e:
            raise RuntimeError(
                f"Resampling required to {target_sr} Hz but librosa is unavailable: {e}"
            )

    return wav.astype(np.float32), sr


def _load_pose_npz(pose_path: Path) -> np.ndarray:
    data = np.load(str(pose_path))

    for key in [
        "joints3d",
        "joints_3d",
        "keypoints3d",
        "pred_3d_joints",
        "positions",
    ]:
        if key in data:
            arr = data[key]
            if arr.ndim == 3 and arr.shape[-1] == 3:
                return np.asarray(arr, dtype=np.float32)

    if "poses" in data and data["poses"].ndim == 2:
        raise ValueError(
            "Found SMPL pose parameters but no 3D joints. Provide a joints3d npz for plotting."
        )

    raise KeyError(
        f"Unsupported pose file format for {pose_path}. Expected a 3D joints array under common keys."
    )


def load_single_sample(
    file_id: str,
    data_root: str | Path = "data/raw/aistpp_hiphop",
    target_sr: int | None = 16000,
) -> Tuple[np.ndarray, Tuple[np.ndarray, int]]:
    root = Path(data_root)
    audio_path = _find_one(
        [
            str(root / "audio" / f"{file_id}.wav"),
            str(root / "audios" / f"{file_id}.wav"),
            str(root / "**" / f"{file_id}.wav"),
        ]
    )
    pose_path = _find_one(
        [
            str(root / "poses" / f"{file_id}.npz"),
            str(root / "keypoints3d" / f"{file_id}.npz"),
            str(root / "**" / f"{file_id}.npz"),
        ]
    )

    pose_seq = _load_pose_npz(pose_path)
    audio_wav, sr = _load_audio(audio_path, target_sr=target_sr)

    return pose_seq, (audio_wav, sr)


def plot_first_frame_3d(pose_seq: np.ndarray) -> None:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    if pose_seq.ndim != 3 or pose_seq.shape[-1] != 3:
        raise ValueError("pose_seq must be (T, J, 3)")

    frame0 = pose_seq[0]
    J = frame0.shape[0]
    connections = [
        (0, 1), (1, 2), (2, 3),
        (0, 4), (4, 5), (5, 6),
        (0, 7), (7, 8), (8, 9),
        (3, 10), (10, 11), (11, 12),
        (3, 13), (13, 14), (14, 15),
        (2, 16), (16, 17),
    ]
    connections = [(a, b) for a, b in connections if a < J and b < J]

    xs, ys, zs = frame0[:, 0], frame0[:, 1], frame0[:, 2]

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xs, ys, zs, c="k", s=8)
    for a, b in connections:
        seg = np.stack([frame0[a], frame0[b]], axis=0)
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], c="tab:blue", linewidth=2)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=10, azim=-70)
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    example_id = "example_file_id"
    try:
        poses, (wav, sr) = load_single_sample(example_id)
        print(poses.shape, wav.shape, sr)
        plot_first_frame_3d(poses)
    except Exception as e:
        print(f"Demo failed: {e}\nUpdate 'example_file_id' to an existing sample.")


