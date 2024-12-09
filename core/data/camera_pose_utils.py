import copy
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


def get_opencv_from_blender(matrix_world, fov, image_size):
    # convert matrix_world to opencv format extrinsics
    opencv_world_to_cam = matrix_world.inverse()
    opencv_world_to_cam[1, :] *= -1
    opencv_world_to_cam[2, :] *= -1
    R, T = opencv_world_to_cam[:3, :3], opencv_world_to_cam[:3, 3]
    R, T = R.unsqueeze(0), T.unsqueeze(0)

    # convert fov to opencv format intrinsics
    focal = 1 / np.tan(fov / 2)
    intrinsics = np.diag(np.array([focal, focal, 1])).astype(np.float32)
    opencv_cam_matrix = torch.from_numpy(intrinsics).unsqueeze(0).float()
    opencv_cam_matrix[:, :2, -1] += torch.tensor([image_size / 2, image_size / 2])
    opencv_cam_matrix[:, [0, 1], [0, 1]] *= image_size / 2

    return R, T, opencv_cam_matrix


def cartesian_to_spherical(xyz):
    xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
    z = np.sqrt(xy + xyz[:, 2] ** 2)
    # for elevation angle defined from z-axis down
    theta = np.arctan2(np.sqrt(xy), xyz[:, 2])
    azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])
    return np.stack([theta, azimuth, z], axis=-1)


def spherical_to_cartesian(spherical_coords):
    # convert from spherical to cartesian coordinates
    theta, azimuth, radius = spherical_coords.T
    x = radius * np.sin(theta) * np.cos(azimuth)
    y = radius * np.sin(theta) * np.sin(azimuth)
    z = radius * np.cos(theta)
    return np.stack([x, y, z], axis=-1)


def look_at(eye, center, up):
    # Create a normalized direction vector from eye to center
    f = np.array(center) - np.array(eye)
    f /= np.linalg.norm(f)

    # Create a normalized right vector
    up_norm = np.array(up) / np.linalg.norm(up)
    s = np.cross(f, up_norm)
    s /= np.linalg.norm(s)

    # Recompute the up vector
    u = np.cross(s, f)

    # Create rotation matrix R
    R = np.array([[s[0], s[1], s[2]], [u[0], u[1], u[2]], [-f[0], -f[1], -f[2]]])

    # Create translation vector T
    T = -np.dot(R, np.array(eye))

    return R, T


def get_blender_from_spherical(elevation, azimuth):
    """Generates blender camera from spherical coordinates."""

    cartesian_coords = spherical_to_cartesian(np.array([[elevation, azimuth, 3.5]]))

    # get camera rotation
    center = np.array([0, 0, 0])
    eye = cartesian_coords[0]
    up = np.array([0, 0, 1])

    R, T = look_at(eye, center, up)
    R = R.T
    T = -np.dot(R, T)
    RT = np.concatenate([R, T.reshape(3, 1)], axis=-1)

    blender_cam = torch.from_numpy(RT).float()
    blender_cam = torch.cat([blender_cam, torch.tensor([[0, 0, 0, 1]])], dim=0)
    print(blender_cam)
    return blender_cam


def invert_pose(r, t):
    r_inv = r.T
    t_inv = -np.dot(r_inv, t)
    return r_inv, t_inv


def transform_pose_sequence_to_relative(poses, as_z_up=False):
    """
    poses: a sequence of 3*4 C2W camera pose matrices
    as_z_up: output in z-up format. If False, the output is in y-up format
    """
    r0, t0 = poses[0][:3, :3], poses[0][:3, 3]
    # r0_inv, t0_inv = invert_pose(r0, t0)
    r0_inv = r0.T
    new_rt0 = np.hstack([np.eye(3, 3), np.zeros((3, 1))])
    if as_z_up:
        new_rt0 = c2w_y_up_to_z_up(new_rt0)
    transformed_poses = [new_rt0]
    for pose in poses[1:]:
        r, t = pose[:3, :3], pose[:3, 3]
        new_r = np.dot(r0_inv, r)
        new_t = np.dot(r0_inv, t - t0)
        new_rt = np.hstack([new_r, new_t[:, None]])
        if as_z_up:
            new_rt = c2w_y_up_to_z_up(new_rt)
        transformed_poses.append(new_rt)
    return transformed_poses


def c2w_y_up_to_z_up(c2w_3x4):
    R_y_up_to_z_up = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

    R = c2w_3x4[:, :3]
    t = c2w_3x4[:, 3]

    R_z_up = R_y_up_to_z_up @ R
    t_z_up = R_y_up_to_z_up @ t

    T_z_up = np.hstack((R_z_up, t_z_up.reshape(3, 1)))

    return T_z_up


def transform_pose_sequence_to_relative_w2c(poses):
    new_rt_list = []
    first_frame_rt = copy.deepcopy(poses[0])
    first_frame_r_inv = first_frame_rt[:, :3].T
    first_frame_t = first_frame_rt[:, -1]
    for rt in poses:
        rt[:, :3] = np.matmul(rt[:, :3], first_frame_r_inv)
        rt[:, -1] = rt[:, -1] - np.matmul(rt[:, :3], first_frame_t)
        new_rt_list.append(copy.deepcopy(rt))
    return new_rt_list


def transform_pose_sequence_to_relative_c2w(poses):
    first_frame_rt = poses[0]
    first_frame_r_inv = first_frame_rt[:, :3].T
    first_frame_t = first_frame_rt[:, -1]
    rotations = poses[:, :, :3]
    translations = poses[:, :, 3]

    # Compute new rotations and translations in batch
    new_rotations = torch.matmul(first_frame_r_inv, rotations)
    new_translations = torch.matmul(
        first_frame_r_inv, (translations - first_frame_t.unsqueeze(0)).unsqueeze(-1)
    )
    # Concatenate new rotations and translations
    new_rt = torch.cat([new_rotations, new_translations], dim=-1)

    return new_rt


def convert_w2c_between_c2w(poses):
    rotations = poses[:, :, :3]
    translations = poses[:, :, 3]
    new_rotations = rotations.transpose(-1, -2)
    new_translations = torch.matmul(-new_rotations, translations.unsqueeze(-1))
    new_rt = torch.cat([new_rotations, new_translations], dim=-1)
    return new_rt


def slerp(q1, q2, t):
    """
    Performs spherical linear interpolation (SLERP) between two quaternions.

    Args:
        q1 (torch.Tensor): Start quaternion (4,).
        q2 (torch.Tensor): End quaternion (4,).
        t (float or torch.Tensor): Interpolation parameter in [0, 1].

    Returns:
        torch.Tensor: Interpolated quaternion (4,).
    """
    q1 = q1 / torch.linalg.norm(q1)  # Normalize q1
    q2 = q2 / torch.linalg.norm(q2)  # Normalize q2

    dot = torch.dot(q1, q2)

    # Ensure shortest path (flip q2 if needed)
    if dot < 0.0:
        q2 = -q2
        dot = -dot

    # Avoid numerical precision issues
    dot = torch.clamp(dot, -1.0, 1.0)

    theta = torch.acos(dot)  # Angle between q1 and q2

    if theta < 1e-6:  # If very close, use linear interpolation
        return (1 - t) * q1 + t * q2

    sin_theta = torch.sin(theta)

    return (torch.sin((1 - t) * theta) / sin_theta) * q1 + (
        torch.sin(t * theta) / sin_theta
    ) * q2


def interpolate_camera_poses(c2w: torch.Tensor, factor: int) -> torch.Tensor:
    """
    Interpolates a sequence of camera c2w poses to N times the length of the original sequence.

    Args:
        c2w (torch.Tensor): Input camera poses of shape (N, 3, 4).
        factor (int): The upsampling factor (e.g., 2 for doubling the length).

    Returns:
        torch.Tensor: Interpolated camera poses of shape (N * factor, 3, 4).
    """
    assert c2w.ndim == 3 and c2w.shape[1:] == (
        3,
        4,
    ), "Input tensor must have shape (N, 3, 4)."
    assert factor > 1, "Upsampling factor must be greater than 1."

    N = c2w.shape[0]
    new_length = N * factor

    # Extract rotations (R) and translations (T)
    rotations = c2w[:, :3, :3]  # Shape (N, 3, 3)
    translations = c2w[:, :3, 3]  # Shape (N, 3)

    # Convert rotations to quaternions for interpolation
    quaternions = torch.tensor(
        R.from_matrix(rotations.numpy()).as_quat()
    )  # Shape (N, 4)

    # Initialize interpolated quaternions and translations
    interpolated_quats = []
    interpolated_translations = []

    # Perform interpolation
    for i in range(N - 1):
        # Start and end quaternions and translations for this segment
        q1, q2 = quaternions[i], quaternions[i + 1]
        t1, t2 = translations[i], translations[i + 1]

        # Time steps for interpolation within this segment
        t_values = torch.linspace(0, 1, factor, dtype=torch.float32)

        # Interpolate quaternions using SLERP
        for t in t_values:
            interpolated_quats.append(slerp(q1, q2, t))

        # Interpolate translations linearly
        interp_t = t1 * (1 - t_values[:, None]) + t2 * t_values[:, None]
        interpolated_translations.append(interp_t)

    interpolated_quats.append(quaternions[0])
    interpolated_translations.append(translations[0].unsqueeze(0))
    # Add the last pose (end of sequence)
    interpolated_quats.append(quaternions[-1])
    interpolated_translations.append(translations[-1].unsqueeze(0))  # Add as 2D tensor

    # Combine interpolated results
    interpolated_quats = torch.stack(interpolated_quats, dim=0)  # Shape (new_length, 4)
    interpolated_translations = torch.cat(
        interpolated_translations, dim=0
    )  # Shape (new_length, 3)

    # Convert quaternions back to rotation matrices
    interpolated_rotations = torch.tensor(
        R.from_quat(interpolated_quats.numpy()).as_matrix()
    )  # Shape (new_length, 3, 3)

    # Form final c2w matrix
    interpolated_c2w = torch.zeros((new_length, 3, 4), dtype=torch.float32)
    interpolated_c2w[:, :3, :3] = interpolated_rotations
    interpolated_c2w[:, :3, 3] = interpolated_translations

    return interpolated_c2w
