"""3D skeleton visualization with camera projection for AIST++ style rendering."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

try:
    from scipy.spatial.transform import Rotation
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def load_camera_params(camera_json_path: Path) -> Dict[str, Dict]:
    """Load camera parameters from JSON file."""
    with open(camera_json_path, "r") as f:
        cameras = json.load(f)
    
    camera_dict = {}
    for cam in cameras:
        name = cam["name"]
        camera_dict[name] = {
            "matrix": np.array(cam["matrix"], dtype=np.float32),
            "distortions": np.array(cam["distortions"], dtype=np.float32),
            "rotation": np.array(cam["rotation"], dtype=np.float32),
            "translation": np.array(cam["translation"], dtype=np.float32),
            "size": tuple(cam["size"]),
        }
    return camera_dict


def rotation_vector_to_matrix(rotation: np.ndarray) -> np.ndarray:
    """Convert rotation vector (axis-angle) to rotation matrix."""
    # rotation is axis-angle representation (3 values)
    angle = np.linalg.norm(rotation)
    if angle < 1e-6:
        return np.eye(3)
    
    if HAS_SCIPY:
        axis = rotation / angle
        return Rotation.from_rotvec(axis * angle).as_matrix()
    else:
        # Manual implementation using Rodriguez formula
        axis = rotation / angle
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        x, y, z = axis
        
        # Rodriguez rotation formula
        R = np.array([
            [cos_a + x*x*(1-cos_a), x*y*(1-cos_a) - z*sin_a, x*z*(1-cos_a) + y*sin_a],
            [y*x*(1-cos_a) + z*sin_a, cos_a + y*y*(1-cos_a), y*z*(1-cos_a) - x*sin_a],
            [z*x*(1-cos_a) - y*sin_a, z*y*(1-cos_a) + x*sin_a, cos_a + z*z*(1-cos_a)]
        ])
        return R


def project_3d_to_2d(
    points_3d: np.ndarray,
    camera_matrix: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    distortions: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D points to 2D image coordinates using camera parameters.
    
    Args:
        points_3d: Array of shape (N, 3) - 3D points in world coordinates
        camera_matrix: 3x3 intrinsic camera matrix
        rotation: 3-element rotation vector (axis-angle)
        translation: 3-element translation vector
        distortions: Distortion coefficients
    
    Returns:
        points_2d: Array of shape (N, 2) - 2D image coordinates
        depths: Array of shape (N,) - depth values for z-ordering
    """
    if points_3d.ndim == 1:
        points_3d = points_3d.reshape(1, -1)
    
    # Convert rotation vector to rotation matrix
    R = rotation_vector_to_matrix(rotation)
    
    # Transform 3D points from world to camera coordinates
    # points_cam = R @ points_3d.T + translation.reshape(3, 1)
    points_cam = (R @ points_3d.T) + translation.reshape(3, 1)
    points_cam = points_cam.T  # Shape: (N, 3)
    
    # Extract depths (z-coordinates in camera frame)
    depths = points_cam[:, 2]
    
    # Project to image plane (z=1 plane)
    x_norm = points_cam[:, 0] / (points_cam[:, 2] + 1e-8)
    y_norm = points_cam[:, 1] / (points_cam[:, 2] + 1e-8)
    
    # Apply distortion (simplified - only radial distortion k1)
    r2 = x_norm**2 + y_norm**2
    k1 = distortions[0] if len(distortions) > 0 else 0.0
    radial = 1 + k1 * r2
    x_dist = x_norm * radial
    y_dist = y_norm * radial
    
    # Apply camera intrinsic matrix
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    
    points_2d = np.zeros((len(points_3d), 2), dtype=np.float32)
    points_2d[:, 0] = fx * x_dist + cx
    points_2d[:, 1] = fy * y_dist + cy
    
    return points_2d, depths


def get_skeleton_structure(num_joints: int = 17) -> Tuple[List[Tuple[int, int]], Dict]:
    """
    Get standard human skeleton hierarchy with proper connections.
    
    Joint hierarchy (AIST++ - based on user observation):
    - Head/Torso: 0, 1, 2, 3, 4 (likely: 0=base, 1=spine1, 2=chest, 3=neck, 4=head)
    - Right arm: 5=shoulder, 7=elbow, 9=wrist
    - Left arm: 6=shoulder, 8=elbow, 10=wrist
    - Right leg: 11=hip, 13=knee, 15=ankle/foot
    - Left leg: 12=hip, 14=knee, 16=ankle/foot
    
    Returns:
        connections: List of (joint_a, joint_b) tuples following standard hierarchy
        colors: Dict mapping (joint_a, joint_b) to color (BGR for OpenCV)
    """
    # Standard human skeleton hierarchy - ONLY anatomical connections
    # Based on user's observation of actual joint positions
    connections = [
        # Head/torso structure: 
        # 4 → 2, 2 → 4 and 0, 0 → 2 and 1, 1 → 0 and 3, 3 → 1
        (4, 2),  # Head (4) -> Chest (2)
        (2, 4),  # Chest (2) -> Head (4) (bidirectional)
        (2, 0),  # Chest (2) -> Base (0)
        (0, 2),  # Base (0) -> Chest (2) (bidirectional)
        (0, 1),  # Base (0) -> Spine1 (1)
        (1, 0),  # Spine1 (1) -> Base (0) (bidirectional)
        (1, 3),  # Spine1 (1) -> Neck (3)
        (3, 1),  # Neck (3) -> Spine1 (1) (bidirectional)
        
        # Shoulders: connect shoulders to each other
        (5, 6),  # Right Shoulder (5) -> Left Shoulder (6)
        # Side connections: shoulders to hips
        (5, 11),  # Right Shoulder (5) -> Right Hip (11)
        (6, 12),  # Left Shoulder (6) -> Left Hip (12)
        
        # Right arm: right shoulder -> right elbow -> right wrist
        (5, 7),  # Right Shoulder (5) -> Right Elbow (7)
        (7, 9),  # Right Elbow (7) -> Right Wrist (9)
        
        # Left arm: left shoulder -> left elbow -> left wrist
        (6, 8),  # Left Shoulder (6) -> Left Elbow (8)
        (8, 10),  # Left Elbow (8) -> Left Wrist (10)
        
        # Right leg: hip -> knee -> ankle
        # Hips now connect through shoulders, not directly from pelvis
        (11, 13),  # Right Hip (11) -> Right Knee (13)
        (13, 15),  # Right Knee (13) -> Right Ankle/Foot (15)
        
        # Left leg: hip -> knee -> ankle
        (12, 14),  # Left Hip (12) -> Left Knee (14)
        (14, 16),  # Left Knee (14) -> Left Ankle/Foot (16)
        
        # Pelvis connection: connect the two hip joints
        (11, 12),  # Right Hip (11) -> Left Hip (12) - Pelvis width
    ]
    # Total: 17 connections - ONLY anatomical, NO cross-body connections
    
    # Optional toes if available
    if num_joints > 17:
        connections.append((13, 17))  # Left Ankle to Left Toe
    if num_joints > 18:
        connections.append((16, 18))  # Right Ankle to Right Toe
    
    # Filter connections based on available joints
    connections = [(a, b) for a, b in connections if a < num_joints and b < num_joints]
    
    # Color scheme (BGR for OpenCV)
    # Torso (spine): Yellow
    # Left arm: Red
    # Right arm: Blue
    # Left leg: Green
    # Right leg: Magenta
    
    colors = {
        # Head/torso structure - Yellow
        (4, 2): (0, 255, 255),  # Yellow (BGR) - Head to Chest
        (2, 4): (0, 255, 255),  # Yellow (BGR) - Chest to Head
        (2, 0): (0, 255, 255),  # Yellow (BGR) - Chest to Base
        (0, 2): (0, 255, 255),  # Yellow (BGR) - Base to Chest
        (0, 1): (0, 255, 255),  # Yellow (BGR) - Base to Spine1
        (1, 0): (0, 255, 255),  # Yellow (BGR) - Spine1 to Base
        (1, 3): (0, 255, 255),  # Yellow (BGR) - Spine1 to Neck
        (3, 1): (0, 255, 255),  # Yellow (BGR) - Neck to Spine1
        
        # Shoulder connections - Yellow (torso color)
        (5, 6): (0, 255, 255),  # Yellow (BGR) - Right Shoulder to Left Shoulder
        # Side connections - Yellow (torso color)
        (5, 11): (0, 255, 255),  # Yellow (BGR) - Right Shoulder to Right Hip
        (6, 12): (0, 255, 255),  # Yellow (BGR) - Left Shoulder to Left Hip
        
        # Right arm - Blue
        (5, 7): (255, 150, 0),  # Blue (BGR) - Right Shoulder to Right Elbow
        (7, 9): (255, 150, 0),  # Right Elbow to Right Wrist
        
        # Left arm - Red
        (6, 8): (0, 100, 255),  # Red (BGR) - Left Shoulder to Left Elbow
        (8, 10): (0, 100, 255),  # Left Elbow to Left Wrist
        
        # Right leg - Magenta
        (11, 13): (255, 0, 255),  # Right Hip to Right Knee
        (13, 15): (255, 0, 255),  # Right Knee to Right Ankle/Foot
        
        # Left leg - Green
        (12, 14): (0, 255, 0),  # Left Hip to Left Knee
        (14, 16): (0, 255, 0),  # Left Knee to Left Ankle/Foot
        
        # Pelvis connection - Yellow (torso color)
        (11, 12): (0, 255, 255),  # Yellow (BGR) - Right Hip to Left Hip (Pelvis width)
    }
    
    # Optional toes
    if num_joints > 17:
        colors[(13, 17)] = (0, 255, 0)  # Left toe - Green
    if num_joints > 18:
        colors[(16, 18)] = (255, 0, 255)  # Right toe - Magenta
    
    return connections, colors


def draw_cylinder_line(
    img: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    """
    Draw a cylinder-like line (bone) between two points.
    Uses anti-aliased line with proper thickness.
    """
    cv2.line(img, pt1, pt2, color, thickness=thickness, lineType=cv2.LINE_AA)


def draw_sphere_joint(
    img: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    color: Tuple[int, int, int],
) -> None:
    """
    Draw a sphere-like joint (filled circle with outline).
    """
    # Draw filled circle (sphere)
    cv2.circle(img, center, radius, color, thickness=-1, lineType=cv2.LINE_AA)
    # Draw dark outline for better visibility
    cv2.circle(img, center, radius, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)


def draw_skeleton_on_image(
    image: np.ndarray,
    points_2d: np.ndarray,
    depths: np.ndarray,
    connections: List[Tuple[int, int]],
    colors: Dict,
    joint_radius: int = 6,
    bone_thickness: int = 4,
    show_joint_labels: bool = False,
) -> np.ndarray:
    """
    Draw skeleton on image with spheres for joints and cylinders for bones.
    Follows standard human joint hierarchy.
    
    Args:
        image: Image array (H, W, 3) in BGR format
        points_2d: Array of shape (J, 2) - 2D joint positions
        depths: Array of shape (J,) - depth values for z-ordering
        connections: List of (joint_a, joint_b) tuples (standard hierarchy)
        colors: Dict mapping connections to BGR colors
        joint_radius: Radius of joint spheres (default: 6)
        bone_thickness: Thickness of bone cylinders (default: 4)
    
    Returns:
        Image with skeleton drawn
    """
    img = image.copy()
    H, W = img.shape[:2]
    
    # Filter out points outside image bounds or invalid
    valid = (
        (points_2d[:, 0] >= 0)
        & (points_2d[:, 0] < W)
        & (points_2d[:, 1] >= 0)
        & (points_2d[:, 1] < H)
        & (depths > 0)  # Only draw joints in front of camera
    )
    
    # Sort connections by average depth (draw farther bones first for proper z-ordering)
    connection_depths = []
    for a, b in connections:
        if a < len(depths) and b < len(depths) and a < len(valid) and b < len(valid):
            if valid[a] and valid[b]:
                avg_depth = (depths[a] + depths[b]) / 2
                connection_depths.append((avg_depth, a, b))
    
    # Sort by depth (farthest first)
    connection_depths.sort(reverse=True)
    
    # Draw bones (cylinders) - farthest first
    for depth, a, b in connection_depths:
        pt_a = tuple(points_2d[a].astype(int))
        pt_b = tuple(points_2d[b].astype(int))
        color = colors.get((a, b), (255, 255, 255))  # Default white
        
        # Draw cylinder-like bone
        draw_cylinder_line(img, pt_a, pt_b, color, thickness=bone_thickness)
    
    # Draw joints (spheres) - farthest first
    joint_depths = [(depths[i], i) for i in range(len(depths)) if valid[i]]
    joint_depths.sort(reverse=True)  # Farthest first
    
    for depth, joint_idx in joint_depths:
        pt = tuple(points_2d[joint_idx].astype(int))
        
        # Joint color - use color from connected bone or default
        color = (255, 255, 255)  # Default white
        for (a, b), bone_color in colors.items():
            if joint_idx == a or joint_idx == b:
                color = bone_color
                break
        
        # Draw sphere-like joint
        draw_sphere_joint(img, pt, joint_radius, color)
        
        # Draw joint label (number) if requested
        if show_joint_labels:
            label = str(joint_idx)
            # Calculate text size and position
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Position label above the joint
            label_x = pt[0] - text_width // 2
            label_y = pt[1] - joint_radius - 5
            
            # Draw background rectangle for better visibility
            cv2.rectangle(
                img,
                (label_x - 2, label_y - text_height - 2),
                (label_x + text_width + 2, label_y + baseline + 2),
                (0, 0, 0),
                -1
            )
            
            # Draw text
            cv2.putText(
                img,
                label,
                (label_x, label_y),
                font,
                font_scale,
                (255, 255, 255),  # White text
                thickness,
                lineType=cv2.LINE_AA,
            )
    
    return img


def overlay_skeleton_on_video_frame(
    frame: np.ndarray,
    pose_3d: np.ndarray,
    camera_params: Dict,
    camera_name: str = "c01",
    joint_radius: int = 5,
    bone_thickness: int = 3,
    show_joint_labels: bool = False,
) -> np.ndarray:
    """
    Overlay 3D skeleton on a single video frame.
    
    Args:
        frame: Video frame (H, W, 3) in BGR format
        pose_3d: Array of shape (17, 3) - 3D joint positions
        camera_params: Camera parameters dict
        camera_name: Name of camera (e.g., "c01")
        joint_radius: Radius of joint spheres
        bone_thickness: Thickness of bone lines
    
    Returns:
        Frame with skeleton overlay
    """
    if camera_name not in camera_params:
        raise ValueError(f"Camera {camera_name} not found in camera parameters")
    
    cam = camera_params[camera_name]
    
    # Project 3D points to 2D
    points_2d, depths = project_3d_to_2d(
        pose_3d,
        cam["matrix"],
        cam["rotation"],
        cam["translation"],
        cam["distortions"],
    )
    
    # Get skeleton structure (with correct number of joints)
    num_joints = pose_3d.shape[0]
    connections, colors = get_skeleton_structure(num_joints=num_joints)
    
    # Draw skeleton
    frame_with_skeleton = draw_skeleton_on_image(
        frame,
        points_2d,
        depths,
        connections,
        colors,
        joint_radius=joint_radius,
        bone_thickness=bone_thickness,
        show_joint_labels=show_joint_labels,
    )
    
    return frame_with_skeleton

