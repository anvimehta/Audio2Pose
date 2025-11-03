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


def plot_stick_figures_2d(
    pose_seq: np.ndarray,
    num_frames: int = 3,
    view: str = "front",
    figsize: tuple = (12, 4),
) -> None:
    """
    Plot 2D stick figures similar to the style in the image.
    
    Args:
        pose_seq: Array of shape (T, J, 3) with 3D pose data
        num_frames: Number of frames to display side by side
        view: Projection view - 'front' (xy), 'side' (xz), or 'top' (yz)
        figsize: Figure size tuple
    """
    import matplotlib.pyplot as plt

    if pose_seq.ndim != 3 or pose_seq.shape[-1] != 3:
        raise ValueError("pose_seq must be (T, J, 3)")

    T, J = pose_seq.shape[:2]
    num_frames = min(num_frames, T)

    # Define skeleton connections (AIST++ 17 joints structure)
    # Match the structure used in the 3D visualization
    connections = [
        (0, 1), (1, 2), (2, 3),  # Spine
        (0, 4), (4, 5), (5, 6),  # Left arm
        (0, 7), (7, 8), (8, 9),  # Right arm
        (3, 10), (10, 11), (11, 12),  # Left leg
        (3, 13), (13, 14), (14, 15),  # Right leg
    ]
    # Optional: additional connections if enough joints
    if J > 16:
        connections.append((2, 16))
    if J > 17:
        connections.append((16, 17))
    connections = [(a, b) for a, b in connections if a < J and b < J]

    # Define segment colors (alternating light blue and light red)
    # Torso/spine: light blue; Limbs alternate between light red and light blue
    segment_colors = {
        # Spine segments - light blue
        (0, 1): "#ADD8E6",  # light blue
        (1, 2): "#ADD8E6",
        (2, 3): "#ADD8E6",
        # Left arm - light red
        (0, 4): "#FFB6C1",  # light red/pink
        (4, 5): "#FFB6C1",
        (5, 6): "#FFB6C1",
        # Right arm - light blue
        (0, 7): "#ADD8E6",
        (7, 8): "#ADD8E6",
        (8, 9): "#ADD8E6",
        # Left leg - light red
        (3, 10): "#FFB6C1",
        (10, 11): "#FFB6C1",
        (11, 12): "#FFB6C1",
        # Right leg - light blue
        (3, 13): "#ADD8E6",
        (13, 14): "#ADD8E6",
        (14, 15): "#ADD8E6",
        # Additional connections
        (2, 16): "#ADD8E6",
        (16, 17): "#ADD8E6",
    }

    # Select frames evenly spaced
    frame_indices = np.linspace(0, T - 1, num_frames, dtype=int)

    # Project 3D to 2D based on view
    if view == "front":
        # Front view: use x and y (depth = z)
        proj_axis = [0, 1]
    elif view == "side":
        # Side view: use z and y (depth = x)
        proj_axis = [2, 1]
    elif view == "top":
        # Top view: use x and z (depth = y)
        proj_axis = [0, 2]
    else:
        raise ValueError(f"Unknown view: {view}. Use 'front', 'side', or 'top'")

    fig, axes = plt.subplots(1, num_frames, figsize=figsize)
    if num_frames == 1:
        axes = [axes]

    for idx, ax in zip(frame_indices, axes):
        frame = pose_seq[idx]
        x_coords = frame[:, proj_axis[0]]
        y_coords = frame[:, proj_axis[1]]

        # Normalize coordinates to center the figure
        x_range = x_coords.max() - x_coords.min()
        y_range = y_coords.max() - y_coords.min()
        x_center = (x_coords.max() + x_coords.min()) / 2
        y_center = (y_coords.max() + y_coords.min()) / 2

        # Draw skeleton connections with colored segments
        for a, b in connections:
            color = segment_colors.get((a, b), "#CCCCCC")  # default gray
            ax.plot(
                [x_coords[a], x_coords[b]],
                [y_coords[a], y_coords[b]],
                color=color,
                linewidth=2.5,
                alpha=0.9,
            )

        # Draw joints (small circles)
        ax.scatter(x_coords, y_coords, c="#333333", s=20, zorder=10, alpha=0.8)

        # Set aspect ratio and remove axes
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_facecolor("white")

        # Add subtle shadow effect (light gray line below)
        shadow_offset = -0.05 * max(x_range, y_range)
        for a, b in connections:
            color = segment_colors.get((a, b), "#CCCCCC")
            ax.plot(
                [x_coords[a], x_coords[b]],
                [y_coords[a] + shadow_offset, y_coords[b] + shadow_offset],
                color="#E0E0E0",
                linewidth=2.5,
                alpha=0.3,
                zorder=0,
            )

        # Set limits with some padding
        padding = 0.15
        x_pad = x_range * padding if x_range > 0 else 0.1
        y_pad = y_range * padding if y_range > 0 else 0.1
        ax.set_xlim(x_center - x_range / 2 - x_pad, x_center + x_range / 2 + x_pad)
        ax.set_ylim(y_center - y_range / 2 - y_pad, y_center + y_range / 2 + y_pad)
        # Note: Y-axis typically represents vertical (height), may need inversion depending on coordinate system
        # Try without inversion first - if person is upside down, remove this comment and uncomment:
        # ax.invert_yaxis()

    plt.tight_layout(pad=0.5)
    plt.show()


def plot_first_frame_2d(pose_seq: np.ndarray, view: str = "front") -> None:
    """Plot the first frame as a 2D stick figure."""
    plot_stick_figures_2d(pose_seq, num_frames=1, view=view, figsize=(4, 4))


def animate_stick_figure_2d(
    pose_seq: np.ndarray,
    view: str = "front",
    fps: int = 30,
    interval: int | None = None,
    save_path: str | Path | None = None,
) -> None:
    """
    Animate a 2D stick figure through the pose sequence.
    
    Args:
        pose_seq: Array of shape (T, J, 3) with 3D pose data
        view: Projection view - 'front' (xy), 'side' (xz), or 'top' (yz)
        fps: Frames per second for animation (default: 30)
        interval: Animation interval in milliseconds (default: 1000/fps)
        save_path: If provided, save animation as GIF/MP4 instead of displaying
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    if pose_seq.ndim != 3 or pose_seq.shape[-1] != 3:
        raise ValueError("pose_seq must be (T, J, 3)")

    T, J = pose_seq.shape[:2]

    # Define skeleton connections
    connections = [
        (0, 1), (1, 2), (2, 3),  # Spine
        (0, 4), (4, 5), (5, 6),  # Left arm
        (0, 7), (7, 8), (8, 9),  # Right arm
        (3, 10), (10, 11), (11, 12),  # Left leg
        (3, 13), (13, 14), (14, 15),  # Right leg
    ]
    if J > 16:
        connections.append((2, 16))
    if J > 17:
        connections.append((16, 17))
    connections = [(a, b) for a, b in connections if a < J and b < J]

    # Define segment colors
    segment_colors = {
        (0, 1): "#ADD8E6", (1, 2): "#ADD8E6", (2, 3): "#ADD8E6",
        (0, 4): "#FFB6C1", (4, 5): "#FFB6C1", (5, 6): "#FFB6C1",
        (0, 7): "#ADD8E6", (7, 8): "#ADD8E6", (8, 9): "#ADD8E6",
        (3, 10): "#FFB6C1", (10, 11): "#FFB6C1", (11, 12): "#FFB6C1",
        (3, 13): "#ADD8E6", (13, 14): "#ADD8E6", (14, 15): "#ADD8E6",
        (2, 16): "#ADD8E6", (16, 17): "#ADD8E6",
    }

    # Project 3D to 2D based on view
    if view == "front":
        proj_axis = [0, 1]
    elif view == "side":
        proj_axis = [2, 1]
    elif view == "top":
        proj_axis = [0, 2]
    else:
        raise ValueError(f"Unknown view: {view}. Use 'front', 'side', or 'top'")

    # Compute bounds across all frames for consistent axis limits
    all_x = pose_seq[:, :, proj_axis[0]]
    all_y = pose_seq[:, :, proj_axis[1]]
    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    padding = 0.15
    x_pad = x_range * padding if x_range > 0 else 0.1
    y_pad = y_range * padding if y_range > 0 else 0.1

    # Setup figure
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("white")

    # Initialize plot elements
    lines = []
    joints_scatter = None
    shadow_lines = []

    def init():
        """Initialize animation."""
        nonlocal joints_scatter
        # Create line objects for skeleton
        for a, b in connections:
            color = segment_colors.get((a, b), "#CCCCCC")
            line, = ax.plot([], [], color=color, linewidth=2.5, alpha=0.9)
            lines.append((line, a, b))
            
            # Shadow lines
            shadow_line, = ax.plot([], [], color="#E0E0E0", linewidth=2.5, alpha=0.3, zorder=0)
            shadow_lines.append(shadow_line)
        
        # Joints scatter
        joints_scatter = ax.scatter([], [], c="#333333", s=20, zorder=10, alpha=0.8)
        
        ax.set_xlim(x_center - x_range / 2 - x_pad, x_center + x_range / 2 + x_pad)
        ax.set_ylim(y_center - y_range / 2 - y_pad, y_center + y_range / 2 + y_pad)
        # Note: Y-axis typically represents vertical (height), may need inversion depending on coordinate system
        # Try without inversion first - if person is upside down, uncomment:
        # ax.invert_yaxis()
        
        return [line for line, _, _ in lines] + [joints_scatter] + shadow_lines

    def animate(frame_idx):
        """Update animation for each frame."""
        if frame_idx >= T:
            return [line for line, _, _ in lines] + [joints_scatter] + shadow_lines
        
        frame = pose_seq[frame_idx]
        x_coords = frame[:, proj_axis[0]]
        y_coords = frame[:, proj_axis[1]]
        
        shadow_offset = -0.05 * max(x_range, y_range)
        
        # Update skeleton lines
        for (line, a, b), shadow_line in zip(lines, shadow_lines):
            line.set_data([x_coords[a], x_coords[b]], [y_coords[a], y_coords[b]])
            shadow_line.set_data(
                [x_coords[a], x_coords[b]],
                [y_coords[a] + shadow_offset, y_coords[b] + shadow_offset]
            )
        
        # Update joints
        joints_scatter.set_offsets(np.column_stack([x_coords, y_coords]))
        
        # Update title with frame number
        ax.set_title(f"Frame {frame_idx + 1}/{T}", fontsize=10, pad=10)
        
        return [line for line, _, _ in lines] + [joints_scatter] + shadow_lines

    # Create animation
    if interval is None:
        interval = int(1000 / fps)  # milliseconds per frame
    
    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=T, interval=interval, blit=True, repeat=True
    )
    
    plt.tight_layout()
    
    if save_path:
        # Save animation
        save_path = Path(save_path)
        if save_path.suffix.lower() == ".gif":
            anim.save(str(save_path), writer="pillow", fps=fps)
        elif save_path.suffix.lower() in [".mp4", ".mov"]:
            anim.save(str(save_path), writer="ffmpeg", fps=fps)
        else:
            raise ValueError(f"Unsupported format: {save_path.suffix}. Use .gif or .mp4")
        print(f"Animation saved to {save_path}")
    else:
        # Display animation
        plt.show()
    
    return anim


def load_pkl(
    file_path: str | Path,
    plot_3d: bool = False,
    plot_2d: bool = True,
    plot_animated: bool = False,
    num_frames_2d: int = 3,
    view_2d: str = "front",
    fps: int = 30,
    save_animation: str | Path | None = None,
) -> dict:
    """
    Load and visualize PKL file with pose data.
    
    Args:
        file_path: Path to PKL file
        plot_3d: Whether to show 3D plot (default: False)
        plot_2d: Whether to show 2D stick figure plot (default: True)
        plot_animated: Whether to show animated 2D stick figure (default: False)
        num_frames_2d: Number of frames to show in 2D plot (default: 3)
        view_2d: 2D projection view - 'front', 'side', or 'top' (default: 'front')
        fps: Frames per second for animation (default: 30)
        save_animation: If provided, save animation to this path instead of displaying
    """
    path = Path(file_path)
    with open(path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise TypeError(f"Expected dict at root of pkl, got {type(data)}")

    _print_shapes(data)

    pose = _find_pose_array(data)
    if pose is not None and pose.ndim == 3 and pose.shape[-1] == 3 and pose.shape[0] > 0:
        if plot_animated:
            animate_stick_figure_2d(pose, view=view_2d, fps=fps, save_path=save_animation)
        elif plot_2d:
            plot_stick_figures_2d(pose, num_frames=num_frames_2d, view=view_2d)
        if plot_3d:
            _plot_first_frame_3d(pose)
    else:
        print("No 3D joints array found for plotting.")

    return data


if __name__ == "__main__":
    import argparse

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


