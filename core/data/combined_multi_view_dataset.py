import PIL
import numpy as np
import torch
from PIL import Image

from .camera_pose_utils import (
    convert_w2c_between_c2w,
    transform_pose_sequence_to_relative_c2w,
)


def get_ray_embeddings(
    poses, size_h=256, size_w=256, fov_xy_list=None, focal_xy_list=None
):
    """
    poses: sequence of cameras poses (y-up format)
    """
    use_focal = False
    if fov_xy_list is None or fov_xy_list[0] is None or fov_xy_list[0][0] is None:
        assert focal_xy_list is not None
        use_focal = True

    rays_embeddings = []
    for i in range(poses.shape[0]):
        cur_pose = poses[i]
        if use_focal:
            rays_o, rays_d = get_rays(
                # [h, w, 3]
                cur_pose,
                size_h,
                size_w,
                focal_xy=focal_xy_list[i],
            )
        else:
            rays_o, rays_d = get_rays(
                cur_pose, size_h, size_w, fov_xy=fov_xy_list[i]
            )  # [h, w, 3]

        rays_plucker = torch.cat(
            [torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1
        )  # [h, w, 6]
        rays_embeddings.append(rays_plucker)

    rays_embeddings = (
        torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous()
    )  # [V, 6, h, w]
    return rays_embeddings


def get_rays(pose, h, w, fov_xy=None, focal_xy=None, opengl=True):
    x, y = torch.meshgrid(
        torch.arange(w, device=pose.device),
        torch.arange(h, device=pose.device),
        indexing="xy",
    )
    x = x.flatten()
    y = y.flatten()

    cx = w * 0.5
    cy = h * 0.5

    # print("fov_xy=", fov_xy)
    # print("focal_xy=", focal_xy)

    if focal_xy is None:
        assert fov_xy is not None, "fov_x/y and focal_x/y cannot both be None."
        focal_x = w * 0.5 / np.tan(0.5 * np.deg2rad(fov_xy[0]))
        focal_y = h * 0.5 / np.tan(0.5 * np.deg2rad(fov_xy[1]))
    else:
        assert (
            len(focal_xy) == 2
        ), "focal_xy should be a list-like object containing only two elements (focal length in x and y direction)."
        focal_x = w * focal_xy[0]
        focal_y = h * focal_xy[1]

    camera_dirs = torch.nn.functional.pad(
        torch.stack(
            [
                (x - cx + 0.5) / focal_x,
                (y - cy + 0.5) / focal_y * (-1.0 if opengl else 1.0),
            ],
            dim=-1,
        ),
        (0, 1),
        value=(-1.0 if opengl else 1.0),
    )  # [hw, 3]

    rays_d = camera_dirs @ pose[:3, :3].transpose(0, 1)  # [hw, 3]
    rays_o = pose[:3, 3].unsqueeze(0).expand_as(rays_d)  # [hw, 3]

    rays_o = rays_o.view(h, w, 3)
    rays_d = safe_normalize(rays_d).view(h, w, 3)

    return rays_o, rays_d


def safe_normalize(x, eps=1e-20):
    return x / length(x, eps)


def length(x, eps=1e-20):
    if isinstance(x, np.ndarray):
        return np.sqrt(np.maximum(np.sum(x * x, axis=-1, keepdims=True), eps))
    else:
        return torch.sqrt(torch.clamp(dot(x, x), min=eps))


def dot(x, y):
    if isinstance(x, np.ndarray):
        return np.sum(x * y, -1, keepdims=True)
    else:
        return torch.sum(x * y, -1, keepdim=True)


def extend_list_by_repeating(original_list, target_length, repeat_idx, at_front):
    if not original_list:
        raise ValueError("The original list cannot be empty.")

    extended_list = []
    original_length = len(original_list)
    for i in range(target_length - original_length):
        extended_list.append(original_list[repeat_idx])

    if at_front:
        extended_list.extend(original_list)
        return extended_list
    else:
        original_list.extend(extended_list)
        return original_list


def select_evenly_spaced_elements(arr, x):
    if x <= 0 or len(arr) == 0:
        return []

    # Calculate step size as the ratio of length of the list and x
    step = len(arr) / x

    # Pick elements at indices that are multiples of step (round them to nearest integer)
    selected_elements = [arr[round(i * step)] for i in range(x)]

    return selected_elements


def convert_co3d_annotation_to_opengl_pose_and_intrinsics(frame_annotation):
    p = frame_annotation.viewpoint.principal_point
    f = frame_annotation.viewpoint.focal_length
    h, w = frame_annotation.image.size
    K = np.eye(3)
    s = (min(h, w) - 1) / 2
    if frame_annotation.viewpoint.intrinsics_format == "ndc_norm_image_bounds":
        K[0, 0] = f[0] * (w - 1) / 2
        K[1, 1] = f[1] * (h - 1) / 2
    elif frame_annotation.viewpoint.intrinsics_format == "ndc_isotropic":
        K[0, 0] = f[0] * s / 2
        K[1, 1] = f[1] * s / 2
    else:
        assert (
            False
        ), f"Invalid intrinsics_format: {frame_annotation.viewpoint.intrinsics_format}"
    K[0, 2] = -p[0] * s + (w - 1) / 2
    K[1, 2] = -p[1] * s + (h - 1) / 2

    R = np.array(frame_annotation.viewpoint.R).T  # note the transpose here
    T = np.array(frame_annotation.viewpoint.T)
    pose = np.concatenate([R, T[:, None]], 1)
    # Need to be converted into OpenGL format. Flip the direction of x, z axis
    pose = np.diag([-1, 1, -1]).astype(np.float32) @ pose
    return pose, K


def normalize_w2c_camera_pose_sequence(
    target_camera_poses,
    condition_camera_poses=None,
    output_c2w=False,
    translation_norm_mode="div_by_max",
):
    """
    Normalize camera pose sequence so that the first frame is identity rotation and zero translation,
    and the translation scale is normalized by the farest point from the first frame (to one).
    :param target_camera_poses: W2C poses tensor in [N, 3, 4]
    :param condition_camera_poses: W2C poses tensor in [N, 3, 4]
    :return: Tuple(Tensor, Tensor), the normalized `target_camera_poses` and `condition_camera_poses`
    """
    # Normalize at w2c, all poses should be in w2c in UnifiedFrame
    num_target_views = target_camera_poses.size(0)
    if condition_camera_poses is not None:
        all_poses = torch.concat([target_camera_poses, condition_camera_poses], dim=0)
    else:
        all_poses = target_camera_poses
    # Convert W2C to C2W
    normalized_poses = transform_pose_sequence_to_relative_c2w(
        convert_w2c_between_c2w(all_poses)
    )
    # Here normalized_poses is C2W
    if not output_c2w:
        # Convert from C2W back to W2C if output_c2w is False.
        normalized_poses = convert_w2c_between_c2w(normalized_poses)

    t_norms = torch.linalg.norm(normalized_poses[:, :, 3], ord=2, dim=-1)
    # print("t_norms=", t_norms)
    largest_t_norm = torch.max(t_norms)

    # print("largest_t_norm=", largest_t_norm)
    # normalized_poses[:, :, 3] -= first_t.unsqueeze(0).repeat(normalized_poses.size(0), 1)
    if translation_norm_mode == "div_by_max_plus_one":
        # Always add a constant component to the translation norm
        largest_t_norm = largest_t_norm + 1.0
    elif translation_norm_mode == "div_by_max":
        largest_t_norm = largest_t_norm
        if largest_t_norm <= 0.05:
            largest_t_norm = 0.05
    elif translation_norm_mode == "disabled":
        largest_t_norm = 1.0
    else:
        assert False, f"Invalid translation_norm_mode: {translation_norm_mode}."
    normalized_poses[:, :, 3] /= largest_t_norm

    target_camera_poses = normalized_poses[:num_target_views]
    if condition_camera_poses is not None:
        condition_camera_poses = normalized_poses[num_target_views:]
    else:
        condition_camera_poses = None
    # print("After First condition:", condition_camera_poses[0])
    # print("After First target:", target_camera_poses[0])
    return target_camera_poses, condition_camera_poses


def central_crop_pil_image(_image, crop_size, use_central_padding=False):
    if use_central_padding:
        # Determine the new size
        _w, _h = _image.size
        new_size = max(_w, _h)
        # Create a new image with white background
        new_image = Image.new("RGB", (new_size, new_size), (255, 255, 255))
        # Calculate the position to paste the original image
        paste_position = ((new_size - _w) // 2, (new_size - _h) // 2)
        # Paste the original image onto the new image
        new_image.paste(_image, paste_position)
        _image = new_image
    # get the new size again if padded
    _w, _h = _image.size
    scale = crop_size / min(_h, _w)
    # resize shortest side to crop_size
    _w_out, _h_out = int(scale * _w), int(scale * _h)
    _image = _image.resize(
        (_w_out, _h_out),
        resample=(
            PIL.Image.Resampling.LANCZOS if scale < 1 else PIL.Image.Resampling.BICUBIC
        ),
    )
    # center crop
    margin_w = (_image.size[0] - crop_size) // 2
    margin_h = (_image.size[1] - crop_size) // 2
    _image = _image.crop(
        (margin_w, margin_h, margin_w + crop_size, margin_h + crop_size)
    )
    return _image


def crop_and_resize(
    image: Image.Image, target_width: int, target_height: int
) -> Image.Image:
    """
    Crops and resizes an image while preserving the aspect ratio.

    Args:
        image (Image.Image): Input PIL image to be cropped and resized.
        target_width (int): Target width of the output image.
        target_height (int): Target height of the output image.

    Returns:
        Image.Image: Cropped and resized image.
    """
    # Original dimensions
    original_width, original_height = image.size
    original_aspect = original_width / original_height
    target_aspect = target_width / target_height

    # Calculate crop box to maintain aspect ratio
    if original_aspect > target_aspect:
        # Crop horizontally
        new_width = int(original_height * target_aspect)
        new_height = original_height
        left = (original_width - new_width) / 2
        top = 0
        right = left + new_width
        bottom = original_height
    else:
        # Crop vertically
        new_width = original_width
        new_height = int(original_width / target_aspect)
        left = 0
        top = (original_height - new_height) / 2
        right = original_width
        bottom = top + new_height

    # Crop and resize
    cropped_image = image.crop((left, top, right, bottom))
    resized_image = cropped_image.resize((target_width, target_height), Image.LANCZOS)

    return resized_image


def calculate_fov_after_resize(
    fov_x: float,
    fov_y: float,
    original_width: int,
    original_height: int,
    target_width: int,
    target_height: int,
) -> (float, float):
    """
    Calculates the new field of view after cropping and resizing an image.

    Args:
        fov_x (float): Original field of view in the x-direction (horizontal).
        fov_y (float): Original field of view in the y-direction (vertical).
        original_width (int): Original width of the image.
        original_height (int): Original height of the image.
        target_width (int): Target width of the output image.
        target_height (int): Target height of the output image.

    Returns:
        (float, float): New field of view (fov_x, fov_y) after cropping and resizing.
    """
    original_aspect = original_width / original_height
    target_aspect = target_width / target_height

    if original_aspect > target_aspect:
        # Crop horizontally
        new_width = int(original_height * target_aspect)
        new_fov_x = fov_x * (new_width / original_width)
        new_fov_y = fov_y
    else:
        # Crop vertically
        new_height = int(original_width / target_aspect)
        new_fov_y = fov_y * (new_height / original_height)
        new_fov_x = fov_x

    return new_fov_x, new_fov_y
