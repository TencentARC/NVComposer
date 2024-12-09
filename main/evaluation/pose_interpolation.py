import torch
import math


def slerp(R1, R2, alpha):
    """
    Perform Spherical Linear Interpolation (SLERP) between two rotation matrices.
    R1, R2: (3x3) rotation matrices.
    alpha: interpolation factor, ranging from 0 to 1.
    """

    # Convert the rotation matrices to quaternions
    def rotation_matrix_to_quaternion(R):
        w = torch.sqrt(1.0 + R[0, 0] + R[1, 1] + R[2, 2]) / 2.0
        w4 = 4.0 * w
        x = (R[2, 1] - R[1, 2]) / w4
        y = (R[0, 2] - R[2, 0]) / w4
        z = (R[1, 0] - R[0, 1]) / w4
        return torch.tensor([w, x, y, z]).float()

    def quaternion_to_rotation_matrix(q):
        w, x, y, z = q
        return torch.tensor(
            [
                [
                    1 - 2 * y * y - 2 * z * z,
                    2 * x * y - 2 * w * z,
                    2 * x * z + 2 * w * y,
                ],
                [
                    2 * x * y + 2 * w * z,
                    1 - 2 * x * x - 2 * z * z,
                    2 * y * z - 2 * w * x,
                ],
                [
                    2 * x * z - 2 * w * y,
                    2 * y * z + 2 * w * x,
                    1 - 2 * x * x - 2 * y * y,
                ],
            ]
        ).float()

    q1 = rotation_matrix_to_quaternion(R1)
    q2 = rotation_matrix_to_quaternion(R2)

    # Dot product of the quaternions
    dot = torch.dot(q1, q2)

    # If the dot product is negative, negate one quaternion to ensure the shortest path is taken
    if dot < 0.0:
        q2 = -q2
        dot = -dot

    # SLERP formula
    if (
        dot > 0.9995
    ):  # If the quaternions are nearly identical, use linear interpolation
        q_interp = (1 - alpha) * q1 + alpha * q2
    else:
        theta_0 = torch.acos(dot)  # Angle between q1 and q2
        sin_theta_0 = torch.sin(theta_0)
        theta = theta_0 * alpha  # Angle between q1 and interpolated quaternion
        sin_theta = torch.sin(theta)
        s1 = torch.sin((1 - alpha) * theta_0) / sin_theta_0
        s2 = sin_theta / sin_theta_0
        q_interp = s1 * q1 + s2 * q2

    # Convert the interpolated quaternion back to a rotation matrix
    R_interp = quaternion_to_rotation_matrix(q_interp)
    return R_interp


def interpolate_camera_poses(pose1, pose2, num_steps):
    """
    Interpolate between two camera poses (3x4 matrices) over a number of steps.

    pose1, pose2: (3x4) camera pose matrices (R|t), where R is a 3x3 rotation matrix and t is a 3x1 translation vector.
    num_steps: number of interpolation steps.

    Returns:
    A list of interpolated poses as (3x4) matrices.
    """
    R1, t1 = pose1[:, :3], pose1[:, 3]
    R2, t2 = pose2[:, :3], pose2[:, 3]

    interpolated_poses = []
    for i in range(num_steps):
        alpha = i / (num_steps - 1)  # Interpolation factor ranging from 0 to 1
        # Interpolate rotation using SLERP
        R_interp = slerp(R1, R2, alpha)
        # Interpolate translation using linear interpolation (LERP)
        t_interp = (1 - alpha) * t1 + alpha * t2
        # Combine interpolated rotation and translation into a (3x4) pose matrix
        pose_interp = torch.cat([R_interp, t_interp.unsqueeze(1)], dim=1)
        interpolated_poses.append(pose_interp)

    return interpolated_poses


def rotation_matrix_from_xyz_angles(x_angle, y_angle, z_angle):
    """
    Compute the rotation matrix from given x, y, z angles (in radians).

    x_angle: Rotation around the x-axis (pitch).
    y_angle: Rotation around the y-axis (yaw).
    z_angle: Rotation around the z-axis (roll).

    Returns:
    A 3x3 rotation matrix.
    """
    # Rotation matrices around each axis
    Rx = torch.tensor(
        [
            [1, 0, 0],
            [0, torch.cos(x_angle), -torch.sin(x_angle)],
            [0, torch.sin(x_angle), torch.cos(x_angle)],
        ]
    ).float()
    Ry = torch.tensor(
        [
            [torch.cos(y_angle), 0, torch.sin(y_angle)],
            [0, 1, 0],
            [-torch.sin(y_angle), 0, torch.cos(y_angle)],
        ]
    ).float()
    Rz = torch.tensor(
        [
            [torch.cos(z_angle), -torch.sin(z_angle), 0],
            [torch.sin(z_angle), torch.cos(z_angle), 0],
            [0, 0, 1],
        ]
    ).float()
    # Combined rotation matrix R = Rz * Ry * Rx
    R_combined = Rz @ Ry @ Rx
    return R_combined.float()


def move_pose(pose1, x_angle, y_angle, z_angle, translation):
    """
    Calculate the second camera pose based on the first pose and given rotations (x, y, z) and translation.

    pose1: The first camera pose (3x4 matrix).
    x_angle, y_angle, z_angle: Rotation angles around the x, y, and z axes, in radians.
    translation: Translation vector (3,).

    Returns:
    pose2: The second camera pose as a (3x4) matrix.
    """
    # Extract the rotation (R1) and translation (t1) from the first pose
    R1 = pose1[:, :3]
    t1 = pose1[:, 3]
    # Calculate the new rotation matrix from the given angles
    R_delta = rotation_matrix_from_xyz_angles(x_angle, y_angle, z_angle)
    # New rotation = R1 * R_delta
    R2 = R1 @ R_delta
    # New translation = t1 + translation
    t2 = t1 + translation
    # Combine R2 and t2 into the new pose (3x4 matrix)
    pose2 = torch.cat([R2, t2.unsqueeze(1)], dim=1)

    return pose2


def deg2rad(degrees):
    """Convert degrees to radians."""
    return degrees * math.pi / 180.0


def generate_spherical_trajectory(end_angles, radius=1.0, num_steps=36):
    """
    Generate a camera-to-world (C2W) trajectory interpolating angles on a sphere.

    Args:
        end_angles (tuple): The endpoint rotation angles in degrees (x, y, z).
                            (start is assumed to be (0, 0, 0)).
        radius (float): Radius of the sphere.
        num_steps (int): Number of steps in the trajectory.

    Returns:
        torch.Tensor: A tensor of shape [num_steps, 3, 4] with the C2W transformations.
    """
    # Convert angles to radians
    end_angles_rad = torch.tensor(
        [deg2rad(angle) for angle in end_angles], dtype=torch.float32
    )
    # Interpolate angles linearly
    interpolated_angles = (
        torch.linspace(0, 1, num_steps).view(-1, 1) * end_angles_rad
    )  # Shape: [num_steps, 3]
    poses = []
    for angles in interpolated_angles:
        # Extract interpolated angles
        x_angle, y_angle = angles
        # Compute camera position on the sphere
        x = radius * math.sin(y_angle) * math.cos(x_angle)
        y = radius * math.sin(x_angle)
        z = radius * math.cos(y_angle) * math.cos(x_angle)
        cam_position = torch.tensor([x, y, z], dtype=torch.float32)
        # Camera's forward direction (looking at the origin)
        look_at_dir = -cam_position / torch.norm(cam_position)
        # Define the "up" vector
        up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
        # Compute the right vector
        right = torch.cross(up, look_at_dir)
        right = right / torch.norm(right)
        # Recompute the orthogonal up vector
        up = torch.cross(look_at_dir, right)
        # Build the rotation matrix
        rotation_matrix = torch.stack([right, up, look_at_dir], dim=0)  # [3, 3]
        # Combine the rotation matrix with the translation (camera position)
        c2w = torch.cat([rotation_matrix, cam_position.view(3, 1)], dim=1)  # [3, 4]
        # Append the pose
        poses.append(c2w)

    return poses
