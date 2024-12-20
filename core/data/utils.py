import copy
import random
from PIL import Image

import numpy as np


def create_relative(RT_list, K_1=4.7, dataset="syn"):
    if dataset == "realestate":
        scale_T = 1
        RT_list = [RT.reshape(3, 4) for RT in RT_list]
    elif dataset == "syn":
        scale_T = (470 / K_1) / 7.5
        """
        4.694746736956946052e+02 0.000000000000000000e+00 4.800000000000000000e+02
        0.000000000000000000e+00 4.694746736956946052e+02 2.700000000000000000e+02
        0.000000000000000000e+00 0.000000000000000000e+00 1.000000000000000000e+00
        """
    elif dataset == "zero123":
        scale_T = 0.5
    else:
        raise Exception("invalid dataset type")

    # convert x y z to x -y -z
    if dataset == "zero123":
        flip_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        for i in range(len(RT_list)):
            RT_list[i] = np.dot(flip_matrix, RT_list[i])

    temp = []
    first_frame_RT = copy.deepcopy(RT_list[0])
    # first_frame_R_inv = np.linalg.inv(first_frame_RT[:,:3])
    first_frame_R_inv = first_frame_RT[:, :3].T
    first_frame_T = first_frame_RT[:, -1]
    for RT in RT_list:
        RT[:, :3] = np.dot(RT[:, :3], first_frame_R_inv)
        RT[:, -1] = RT[:, -1] - np.dot(RT[:, :3], first_frame_T)
        RT[:, -1] = RT[:, -1] * scale_T
        temp.append(RT)
    RT_list = temp

    if dataset == "realestate":
        RT_list = [RT.reshape(-1) for RT in RT_list]

    return RT_list


def sigma_matrix2(sig_x, sig_y, theta):
    """Calculate the rotated sigma matrix (two dimensional matrix).
    Args:
        sig_x (float):
        sig_y (float):
        theta (float): Radian measurement.
    Returns:
        ndarray: Rotated sigma matrix.
    """
    d_matrix = np.array([[sig_x**2, 0], [0, sig_y**2]])
    u_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    return np.dot(u_matrix, np.dot(d_matrix, u_matrix.T))


def mesh_grid(kernel_size):
    """Generate the mesh grid, centering at zero.
    Args:
        kernel_size (int):
    Returns:
        xy (ndarray): with the shape (kernel_size, kernel_size, 2)
        xx (ndarray): with the shape (kernel_size, kernel_size)
        yy (ndarray): with the shape (kernel_size, kernel_size)
    """
    ax = np.arange(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
    xx, yy = np.meshgrid(ax, ax)
    xy = np.hstack(
        (
            xx.reshape((kernel_size * kernel_size, 1)),
            yy.reshape(kernel_size * kernel_size, 1),
        )
    ).reshape(kernel_size, kernel_size, 2)
    return xy, xx, yy


def pdf2(sigma_matrix, grid):
    """Calculate PDF of the bivariate Gaussian distribution.
    Args:
        sigma_matrix (ndarray): with the shape (2, 2)
        grid (ndarray): generated by :func:`mesh_grid`,
            with the shape (K, K, 2), K is the kernel size.
    Returns:
        kernel (ndarrray): un-normalized kernel.
    """
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.exp(-0.5 * np.sum(np.dot(grid, inverse_sigma) * grid, 2))
    return kernel


def bivariate_Gaussian(kernel_size, sig_x, sig_y, theta, grid=None, isotropic=True):
    """Generate a bivariate isotropic or anisotropic Gaussian kernel.
    In the isotropic mode, only `sig_x` is used. `sig_y` and `theta` is ignored.
    Args:
        kernel_size (int):
        sig_x (float):
        sig_y (float):
        theta (float): Radian measurement.
        grid (ndarray, optional): generated by :func:`mesh_grid`,
            with the shape (K, K, 2), K is the kernel size. Default: None
        isotropic (bool):
    Returns:
        kernel (ndarray): normalized kernel.
    """
    if grid is None:
        grid, _, _ = mesh_grid(kernel_size)
    if isotropic:
        sigma_matrix = np.array([[sig_x**2, 0], [0, sig_x**2]])
    else:
        sigma_matrix = sigma_matrix2(sig_x, sig_y, theta)
    kernel = pdf2(sigma_matrix, grid)
    kernel = kernel / np.sum(kernel)
    return kernel


def rgba_to_rgb_with_bg(rgba_image, bg_color=(255, 255, 255)):
    """
    Convert a PIL RGBA Image to an RGB Image with a white background.

    Args:
    rgba_image (Image): A PIL Image object in RGBA mode.

    Returns:
    Image: A PIL Image object in RGB mode with white background.
    """
    # Ensure the image is in RGBA mode
    # Ensure the image is in RGBA mode
    if rgba_image.mode != "RGBA":
        return rgba_image
        # raise ValueError("The image must be in RGBA mode")

    # Create a white background image
    white_bg_rgb = Image.new("RGB", rgba_image.size, bg_color)
    # Paste the RGBA image onto the white background using alpha channel as mask
    white_bg_rgb.paste(
        rgba_image, mask=rgba_image.split()[3]
    )  # 3 is the alpha channel index
    return white_bg_rgb


def random_order_preserving_selection(items, num):
    if num > len(items):
        print("WARNING: Item list is shorter than `num` given.")
        return items
    selected_indices = sorted(random.sample(range(len(items)), num))
    selected_items = [items[i] for i in selected_indices]
    return selected_items


def pad_pil_image_to_square(image, fill_color=(255, 255, 255)):
    """
    Pad an image to make it square with the given fill color.

    Args:
    image (PIL.Image): The original image.
    fill_color (tuple): The color to use for padding (default is black).

    Returns:
    PIL.Image: A new image that is padded to be square.
    """
    width, height = image.size

    # Determine the new size, which will be the maximum of width or height
    new_size = max(width, height)

    # Create a new image with the new size and fill color
    new_image = Image.new("RGB", (new_size, new_size), fill_color)

    # Calculate the position to paste the original image onto the new image
    # This calculation centers the original image in the new square canvas
    left = (new_size - width) // 2
    top = (new_size - height) // 2

    # Paste the original image into the new image
    new_image.paste(image, (left, top))

    return new_image
