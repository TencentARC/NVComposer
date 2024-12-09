import datetime
import json
import os

import gradio as gr
from huggingface_hub import hf_hub_download
import PIL.Image
import numpy as np
import torch
import torchvision.transforms.functional
from numpy import deg2rad
from omegaconf import OmegaConf

from core.data.camera_pose_utils import convert_w2c_between_c2w
from core.data.combined_multi_view_dataset import (
    get_ray_embeddings,
    normalize_w2c_camera_pose_sequence,
    crop_and_resize,
)
from main.evaluation.funcs import load_model_checkpoint
from main.evaluation.pose_interpolation import (
    move_pose,
    interpolate_camera_poses,
    generate_spherical_trajectory,
)
from main.evaluation.utils_eval import process_inference_batch
from utils.utils import instantiate_from_config
from core.models.samplers.ddim import DDIMSampler

torch.set_float32_matmul_precision("medium")

gpu_no = 0
config = "./configs/dual_stream/nvcomposer.yaml"
ckpt = hf_hub_download(repo_id="TencentARC/NVComposer", filename="NVComposer-V0.1.ckpt")

model_resolution_height, model_resolution_width = 576, 1024
num_views = 16
dtype = torch.float16
config = OmegaConf.load(config)
model_config = config.pop("model", OmegaConf.create())
model_config.params.train_with_multi_view_feature_alignment = False
model = instantiate_from_config(model_config).cuda(gpu_no).to(dtype=dtype)
assert os.path.exists(ckpt), f"Error: checkpoint [{ckpt}] Not Found!"
print(f"Loading checkpoint from {ckpt}...")
model = load_model_checkpoint(model, ckpt)
model.eval()
latent_h, latent_w = (
    model_resolution_height // 8,
    model_resolution_width // 8,
)
channels = model.channels
sampler = DDIMSampler(model)

EXAMPLES = [
    [
        "./assets/sample1.jpg",
        None,
        1,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        -0.2,
        3,
        1.5,
        20,
        "./assets/sample1.mp4",
        1,
    ],
    [
        "./assets/sample2.jpg",
        None,
        0,
        0,
        25,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        3,
        1.5,
        20,
        "./assets/sample2.mp4",
        1,
    ],
    [
        "./assets/sample3.jpg",
        None,
        0,
        0,
        15,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        3,
        1.5,
        20,
        "./assets/sample3.mp4",
        1,
    ],
    [
        "./assets/sample4.jpg",
        None,
        0,
        0,
        -15,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        3,
        1.5,
        20,
        "./assets/sample4.mp4",
        1,
    ],
    [
        "./assets/sample5-1.png",
        "./assets/sample5-2.png",
        0,
        0,
        -30,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        3,
        1.5,
        20,
        "./assets/sample5.mp4",
        2,
    ],
]


def compose_data_item(
    num_views,
    cond_pil_image_list,
    caption="",
    camera_mode=False,
    input_pose_format="c2w",
    model_pose_format="c2w",
    x_rotation_angle=10,
    y_rotation_angle=10,
    z_rotation_angle=10,
    x_translation=0.5,
    y_translation=0.5,
    z_translation=0.5,
    image_size=None,
    spherical_angle_x=10,
    spherical_angle_y=10,
    spherical_radius=10,
):
    if image_size is None:
        image_size = [512, 512]
    latent_size = [image_size[0] // 8, image_size[1] // 8]

    def image_processing_function(x):
        return (
            torch.from_numpy(
                np.array(
                    crop_and_resize(
                        x, target_height=image_size[0], target_width=image_size[1]
                    )
                ).transpose((2, 0, 1))
            ).float()
            / 255.0
        )

    resizer_image_to_latent_size = torchvision.transforms.Resize(
        size=latent_size,
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        antialias=True,
    )
    num_cond_views = len(cond_pil_image_list)
    print(f"Number of received condition images: {num_cond_views}.")
    num_target_views = num_views - num_cond_views
    if camera_mode == 1:
        print("Camera Mode: Movement with Rotation and Translation.")
        start_pose = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
            ]
        ).float()
        end_pose = move_pose(
            start_pose,
            x_angle=torch.tensor(deg2rad(x_rotation_angle)),
            y_angle=torch.tensor(deg2rad(y_rotation_angle)),
            z_angle=torch.tensor(deg2rad(z_rotation_angle)),
            translation=torch.tensor([x_translation, y_translation, z_translation]),
        )
        target_poses = interpolate_camera_poses(
            start_pose, end_pose, num_steps=num_target_views
        )
    elif camera_mode == 0:
        print("Camera Mode: Spherical Movement.")
        target_poses = generate_spherical_trajectory(
            end_angles=(spherical_angle_x, spherical_angle_y),
            radius=spherical_radius,
            num_steps=num_target_views,
        )
    print("Target pose sequence (before normalization): \n  ", target_poses)
    cond_poses = [
        torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
            ]
        ).float()
    ] * num_cond_views
    target_poses = torch.stack(target_poses, dim=0).float()
    cond_poses = torch.stack(cond_poses, dim=0).float()
    if not camera_mode != 0 and (input_pose_format != "w2c"):
        # c2w to w2c. Input for normalize_camera_pose_sequence() should be w2c
        target_poses = convert_w2c_between_c2w(target_poses)
        cond_poses = convert_w2c_between_c2w(cond_poses)
    target_poses, cond_poses = normalize_w2c_camera_pose_sequence(
        target_poses,
        cond_poses,
        output_c2w=model_pose_format == "c2w",
        translation_norm_mode="disabled",
    )
    target_and_condition_camera_poses = torch.cat([target_poses, cond_poses], dim=0)

    print("Target pose sequence (after normalization): \n  ", target_poses)
    fov_xy = [80, 45]
    target_rays = get_ray_embeddings(
        target_poses,
        size_h=image_size[0],
        size_w=image_size[1],
        fov_xy_list=[fov_xy for _ in range(num_target_views)],
    )
    condition_rays = get_ray_embeddings(
        cond_poses,
        size_h=image_size[0],
        size_w=image_size[1],
        fov_xy_list=[fov_xy for _ in range(num_cond_views)],
    )
    target_images_tensor = torch.zeros(
        num_target_views, 3, image_size[0], image_size[1]
    )
    condition_images = [image_processing_function(x) for x in cond_pil_image_list]
    condition_images_tensor = torch.stack(condition_images, dim=0) * 2.0 - 1.0
    target_images_tensor[0, :, :, :] = condition_images_tensor[0, :, :, :]
    target_and_condition_images_tensor = torch.cat(
        [target_images_tensor, condition_images_tensor], dim=0
    )
    target_and_condition_rays_tensor = torch.cat([target_rays, condition_rays], dim=0)
    target_and_condition_rays_tensor = resizer_image_to_latent_size(
        target_and_condition_rays_tensor * 5.0
    )
    mask_preserving_target = torch.ones(size=[num_views, 1], dtype=torch.float16)
    mask_preserving_target[num_target_views:] = 0.0
    combined_fovs = torch.stack([torch.tensor(fov_xy)] * num_views, dim=0)

    mask_only_preserving_first_target = torch.zeros_like(mask_preserving_target)
    mask_only_preserving_first_target[0] = 1.0
    mask_only_preserving_first_condition = torch.zeros_like(mask_preserving_target)
    mask_only_preserving_first_condition[num_target_views] = 1.0
    test_data = {
        # T, C, H, W
        "combined_images": target_and_condition_images_tensor.unsqueeze(0),
        "mask_preserving_target": mask_preserving_target.unsqueeze(0),  # T, 1
        # T, 1
        "mask_only_preserving_first_target": mask_only_preserving_first_target.unsqueeze(
            0
        ),
        # T, 1
        "mask_only_preserving_first_condition": mask_only_preserving_first_condition.unsqueeze(
            0
        ),
        # T, C, H//8, W//8
        "combined_rays": target_and_condition_rays_tensor.unsqueeze(0),
        "combined_fovs": combined_fovs.unsqueeze(0),
        "target_and_condition_camera_poses": target_and_condition_camera_poses.unsqueeze(
            0
        ),
        "num_target_images": torch.tensor([num_target_views]),
        "num_cond_images": torch.tensor([num_cond_views]),
        "num_cond_images_str": [str(num_cond_views)],
        "item_idx": [0],
        "subset_key": ["evaluation"],
        "caption": [caption],
        "fov_xy": torch.tensor(fov_xy).float().unsqueeze(0),
    }
    return test_data


def tensor_to_mp4(video, savepath, fps, nrow=None):
    """
    video: torch.Tensor, b,t,c,h,w,  value range: 0-1
    """
    n = video.shape[0]
    print("Video shape=", video.shape)
    video = video.permute(1, 0, 2, 3, 4)  # t,n,c,h,w
    nrow = int(np.sqrt(n)) if nrow is None else nrow
    frame_grids = [
        torchvision.utils.make_grid(framesheet, nrow=nrow) for framesheet in video
    ]  # [3, grid_h, grid_w]
    # stack in temporal dim [T, 3, grid_h, grid_w]
    grid = torch.stack(frame_grids, dim=0)
    grid = torch.clamp(grid.float(), -1.0, 1.0)
    # [T, 3, grid_h, grid_w] -> [T, grid_h, grid_w, 3]
    grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)
    # print(f'Save video to {savepath}')
    torchvision.io.write_video(
        savepath, grid, fps=fps, video_codec="h264", options={"crf": "10"}
    )


def parse_to_np_array(input_string):
    try:
        # Try to parse the input as JSON first
        data = json.loads(input_string)
        arr = np.array(data)
    except json.JSONDecodeError:
        # If JSON parsing fails, assume it's a multi-line string and handle accordingly
        lines = input_string.strip().splitlines()
        data = []
        for line in lines:
            # Split the line by spaces and convert to floats
            data.append([float(x) for x in line.split()])
        arr = np.array(data)

    # Check if the resulting array is 3x4
    if arr.shape != (3, 4):
        raise ValueError(f"Expected array shape (3, 4), but got {arr.shape}")

    return arr


def run_inference(
    camera_mode,
    input_cond_image1=None,
    input_cond_image2=None,
    input_cond_image3=None,
    input_cond_image4=None,
    input_pose_format="c2w",
    model_pose_format="c2w",
    x_rotation_angle=None,
    y_rotation_angle=None,
    z_rotation_angle=None,
    x_translation=None,
    y_translation=None,
    z_translation=None,
    trajectory_extension_factor=1,
    cfg_scale=1.0,
    cfg_scale_extra=1.0,
    sample_steps=50,
    num_images_slider=None,
    spherical_angle_x=10,
    spherical_angle_y=10,
    spherical_radius=10,
    random_seed=1,
):
    cfg_scale_extra = 1.0  # Disable Extra CFG due to time limit of ZeroGPU
    os.makedirs("./cache/", exist_ok=True)
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            torch.manual_seed(random_seed)
            input_cond_images = []
            for _cond_image in [
                input_cond_image1,
                input_cond_image2,
                input_cond_image3,
                input_cond_image4,
            ]:
                if _cond_image is not None:
                    if isinstance(_cond_image, np.ndarray):
                        _cond_image = PIL.Image.fromarray(_cond_image)
                    input_cond_images.append(_cond_image)
            num_condition_views = len(input_cond_images)
            assert (
                num_images_slider == num_condition_views
            ), f"The `num_condition_views`={num_condition_views} while got `num_images_slider`={num_images_slider}."
            input_caption = ""
            num_target_views = num_views - num_condition_views
            data_item = compose_data_item(
                num_views=num_views,
                cond_pil_image_list=input_cond_images,
                caption=input_caption,
                camera_mode=camera_mode,
                input_pose_format=input_pose_format,
                model_pose_format=model_pose_format,
                x_rotation_angle=x_rotation_angle,
                y_rotation_angle=y_rotation_angle,
                z_rotation_angle=z_rotation_angle,
                x_translation=x_translation,
                y_translation=y_translation,
                z_translation=z_translation,
                image_size=[model_resolution_height, model_resolution_width],
                spherical_angle_x=spherical_angle_x,
                spherical_angle_y=spherical_angle_y,
                spherical_radius=spherical_radius,
            )
            batch = data_item
            if trajectory_extension_factor == 1:
                print("No trajectory extension.")
            else:
                print(f"Trajectory is enabled: {trajectory_extension_factor}.")
            full_x_samples = []
            for repeat_idx in range(int(trajectory_extension_factor)):
                if repeat_idx != 0:
                    batch["combined_images"][:, 0, :, :, :] = full_x_samples[-1][
                        :, -1, :, :, :
                    ]
                    batch["combined_images"][:, num_target_views, :, :, :] = (
                        full_x_samples[-1][:, -1, :, :, :]
                    )
                cond, uc, uc_extra, x_rec = process_inference_batch(
                    cfg_scale, batch, model, with_uncondition_extra=True
                )

                batch_size = x_rec.shape[0]
                shape_without_batch = (num_views, channels, latent_h, latent_w)
                samples, _ = sampler.sample(
                    sample_steps,
                    batch_size=batch_size,
                    shape=shape_without_batch,
                    conditioning=cond,
                    verbose=True,
                    unconditional_conditioning=uc,
                    unconditional_guidance_scale=cfg_scale,
                    unconditional_conditioning_extra=uc_extra,
                    unconditional_guidance_scale_extra=cfg_scale_extra,
                    x_T=None,
                    expand_mode=False,
                    num_target_views=num_views - num_condition_views,
                    num_condition_views=num_condition_views,
                    dense_expansion_ratio=None,
                    pred_x0_post_process_function=None,
                    pred_x0_post_process_function_kwargs=None,
                )

                if samples.size(2) > 4:
                    image_samples = samples[:, :num_target_views, :4, :, :]
                else:
                    image_samples = samples
                per_instance_decoding = False
                if per_instance_decoding:
                    x_samples = []
                    for item_idx in range(image_samples.shape[0]):
                        image_samples = image_samples[
                            item_idx : item_idx + 1, :, :, :, :
                        ]
                        x_sample = model.decode_first_stage(image_samples)
                        x_samples.append(x_sample)
                    x_samples = torch.cat(x_samples, dim=0)
                else:
                    x_samples = model.decode_first_stage(image_samples)
                full_x_samples.append(x_samples[:, :num_target_views, ...])

            full_x_samples = torch.concat(full_x_samples, dim=1)
            x_samples = full_x_samples
            x_samples = torch.clamp((x_samples + 1.0) / 2.0, 0.0, 1.0)
            video_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".mp4"
            video_path = "./cache/" + video_name
            tensor_to_mp4(x_samples.detach().cpu(), fps=6, savepath=video_path)
            return video_path


with gr.Blocks() as demo:
    gr.HTML(
        """
<div style="text-align: center;">
    <h1 style="text-align: center; color: #333333;">📸 NVComposer</h1>
    <h3 style="text-align: center; color: #333333;">Generative Novel View Synthesis with Sparse and
    Unposed Images</h3>
    <p style="text-align: center; font-weight: bold">
        <a href="https://lg-li.github.io/project/nvcomposer">Project Page</a> | 
        <a href="https://arxiv.org/abs/2412.03517">ArXiv Preprint</a> | 
        <a href="https://github.com/TencentARC/NVComposer">Github Repository</a>
    </p>
    <p style="text-align: left; font-size: 1.1em;">
    Welcome to the demo of <strong>NVComposer</strong>. Follow the steps below to explore the capabilities
    of our model:
    </p>
</div>
<div style="text-align: left; margin: 0 auto; ">
    <ol style="font-size: 1.1em;">
    <li><strong>Choose camera movement mode:</strong> Spherical Mode or Rotation & Translation Mode.</li>
    <li><strong>Customize the camera trajectory:</strong> Adjust the spherical parameters or rotation/translations along the X, Y,
        and Z axes.</li>
    <li><strong>Upload images:</strong> You can upload up to 4 images as input conditions.</li>
    <li><strong>Set sampling parameters (optional):</strong> Tweak the settings and click the <b>Generate</b> button.</li>
    </ol>
    <p>
    ⏱️ <b>ZeroGPU Time Limit</b>: Hugging Face ZeroGPU has a inference time limit of 180 seconds.
    You may need to <b>log in with a free account</b> to use this demo.
    Large sampling steps might lead to timeout (GPU Abort).
    In that case, please consider log in with a Pro account or run it on your local machine. 
    </p>
</div>
    """
    )
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Accordion("Camera Movement Settings", open=True):
                camera_mode = gr.Radio(
                    choices=[("Spherical Mode", 0), ("Rotation & Translation Mode", 1)],
                    label="Camera Mode",
                    value=0,
                    interactive=True,
                )

                with gr.Group(visible=True) as group_spherical:
                    # This tab can be left blank for now as per your request
                    # Add extra options manually here in the future
                    gr.HTML(
                        """<p style="padding: 10px">
                            <b>Spherical Mode</b> allows you to control the camera's movement by specifying its position on a sphere centered around the scene.
                            Adjust the Polar Angle (vertical rotation), Azimuth Angle (horizontal rotation), and Radius (distance from the center of the anchor view) to define the camera's viewpoint.
                            The anchor view is considered located on the sphere at the specified radius, aligned with a zero polar angle and zero azimuth angle, oriented toward the origin.
                            </p>
                            """
                    )
                    spherical_angle_x = gr.Slider(
                        minimum=-30,
                        maximum=30,
                        step=1,
                        value=0,
                        label="Polar Angle (Theta)",
                    )
                    spherical_angle_y = gr.Slider(
                        minimum=-30,
                        maximum=30,
                        step=1,
                        value=5,
                        label="Azimuth Angle (Phi)",
                    )
                    spherical_radius = gr.Slider(
                        minimum=0.5, maximum=1.5, step=0.1, value=1, label="Radius"
                    )

                with gr.Group(visible=False) as group_move_rotation_translation:
                    gr.HTML(
                        """<p style="padding: 10px">
                            <b>Rotation & Translation Mode</b> lets you directly define how the camera moves and rotates in the 3D space.
                            Use Rotation X/Y/Z to control the camera's orientation and Translation X/Y/Z to shift its position.
                            The anchor view serves as the starting point, with no initial rotation or translation applied.
                            </p>
                            """
                    )
                    rotation_x = gr.Slider(
                        minimum=-20, maximum=20, step=1, value=0, label="Rotation X"
                    )
                    rotation_y = gr.Slider(
                        minimum=-20, maximum=20, step=1, value=0, label="Rotation Y"
                    )
                    rotation_z = gr.Slider(
                        minimum=-20, maximum=20, step=1, value=0, label="Rotation Z"
                    )
                    translation_x = gr.Slider(
                        minimum=-1, maximum=1, step=0.1, value=0, label="Translation X"
                    )
                    translation_y = gr.Slider(
                        minimum=-1, maximum=1, step=0.1, value=0, label="Translation Y"
                    )
                    translation_z = gr.Slider(
                        minimum=-1,
                        maximum=1,
                        step=0.1,
                        value=-0.2,
                        label="Translation Z",
                    )

                input_camera_pose_format = gr.Radio(
                    choices=["W2C", "C2W"],
                    value="C2W",
                    label="Input Camera Pose Format",
                    visible=False,
                )
                model_camera_pose_format = gr.Radio(
                    choices=["W2C", "C2W"],
                    value="C2W",
                    label="Model Camera Pose Format",
                    visible=False,
                )

                def on_change_selected_camera_settings(_id):
                    return [gr.update(visible=_id == 0), gr.update(visible=_id == 1)]

                camera_mode.change(
                    fn=on_change_selected_camera_settings,
                    inputs=camera_mode,
                    outputs=[group_spherical, group_move_rotation_translation],
                )

            with gr.Accordion("Advanced Sampling Settings"):
                cfg_scale = gr.Slider(
                    value=3.0,
                    label="Classifier-Free Guidance Scale",
                    minimum=1,
                    maximum=10,
                    step=0.1,
                )
                extra_cfg_scale = gr.Slider(
                    value=1.0,
                    label="Extra Classifier-Free Guidance Scale",
                    minimum=1,
                    maximum=10,
                    step=0.1,
                    visible=False,
                )
                sample_steps = gr.Slider(
                    value=18, label="DDIM Sample Steps", minimum=0, maximum=25, step=1
                )
                trajectory_extension_factor = gr.Slider(
                    value=1,
                    label="Trajectory Extension (proportional to runtime)",
                    minimum=1,
                    maximum=3,
                    step=1,
                )
                random_seed = gr.Slider(
                    value=1024, minimum=1, maximum=9999, step=1, label="Random Seed"
                )

                def on_change_trajectory_extension_factor(_val):
                    if _val == 1:
                        return [
                            gr.update(minimum=-30, maximum=30),
                            gr.update(minimum=-30, maximum=30),
                            gr.update(minimum=0.5, maximum=1.5),
                            gr.update(minimum=-20, maximum=20),
                            gr.update(minimum=-20, maximum=20),
                            gr.update(minimum=-20, maximum=20),
                            gr.update(minimum=-1, maximum=1),
                            gr.update(minimum=-1, maximum=1),
                            gr.update(minimum=-1, maximum=1),
                        ]
                    elif _val == 2:
                        return [
                            gr.update(minimum=-15, maximum=15),
                            gr.update(minimum=-15, maximum=15),
                            gr.update(minimum=0.5, maximum=1.5),
                            gr.update(minimum=-10, maximum=10),
                            gr.update(minimum=-10, maximum=10),
                            gr.update(minimum=-10, maximum=10),
                            gr.update(minimum=-0.5, maximum=0.5),
                            gr.update(minimum=-0.5, maximum=0.5),
                            gr.update(minimum=-0.5, maximum=0.5),
                        ]
                    elif _val == 3:
                        return [
                            gr.update(minimum=-10, maximum=10),
                            gr.update(minimum=-10, maximum=10),
                            gr.update(minimum=0.5, maximum=1.5),
                            gr.update(minimum=-6, maximum=6),
                            gr.update(minimum=-6, maximum=6),
                            gr.update(minimum=-6, maximum=6),
                            gr.update(minimum=-0.3, maximum=0.3),
                            gr.update(minimum=-0.3, maximum=0.3),
                            gr.update(minimum=-0.3, maximum=0.3),
                        ]

                trajectory_extension_factor.change(
                    fn=on_change_trajectory_extension_factor,
                    inputs=trajectory_extension_factor,
                    outputs=[
                        spherical_angle_x,
                        spherical_angle_y,
                        spherical_radius,
                        rotation_x,
                        rotation_y,
                        rotation_z,
                        translation_x,
                        translation_y,
                        translation_z,
                    ],
                )

        with gr.Column(scale=1):
            with gr.Accordion("Input Image(s)", open=True):
                num_images_slider = gr.Slider(
                    minimum=1,
                    maximum=4,
                    step=1,
                    value=1,
                    label="Number of Input Image(s)",
                )
                condition_image_1 = gr.Image(label="Input Image 1 (Anchor View)")
                condition_image_2 = gr.Image(label="Input Image 2", visible=False)
                condition_image_3 = gr.Image(label="Input Image 3", visible=False)
                condition_image_4 = gr.Image(label="Input Image 4", visible=False)

        with gr.Column(scale=1):
            with gr.Accordion("Output Video", open=True):
                output_video = gr.Video(label="Output Video")
            run_btn = gr.Button("Generate")
            with gr.Accordion("Notes", open=True):
                gr.HTML(
                    """
<p style="font-size: 1.1em; line-height: 1.6; color: #555;">
🧐 <b>Reminder</b>:
    As a generative model, NVComposer may occasionally produce unexpected outputs.
    Try adjusting the random seed, sampling steps, or CFG scales to explore different results.
<br>
🤔 <b>Longer Generation</b>:
    If you need longer video, you can increase the trajectory extension value in the advanced sampling settings and run with your own GPU.
    This extends the defined camera trajectory by repeating it, allowing for a longer output.
    This also requires using smaller rotation or translation scales to maintain smooth transitions and will increase the generation time.  <br>
🤗 <b>Limitation</b>:
    This is the initial beta version of NVComposer, and its generalizability may be limited in certain scenarios (e.g., human).
    We’re actively working on an improved version with enhanced datasets and a more powerful foundation model,
    and we are looking for <b>collaboration opportunities from the community</b>. <br>
✨ We welcome your feedback and questions. Thank you! </p>
                """
                )

    with gr.Row():
        gr.Examples(
            label="Quick Examples",
            examples=EXAMPLES,
            inputs=[
                condition_image_1,
                condition_image_2,
                camera_mode,
                spherical_angle_x,
                spherical_angle_y,
                spherical_radius,
                rotation_x,
                rotation_y,
                rotation_z,
                translation_x,
                translation_y,
                translation_z,
                cfg_scale,
                extra_cfg_scale,
                sample_steps,
                output_video,
                num_images_slider,
            ],
            examples_per_page=5,
            cache_examples=False,
        )

    # Update visibility of condition images based on the slider
    def update_visible_images(num_images):
        return [
            gr.update(visible=num_images >= 2),
            gr.update(visible=num_images >= 3),
            gr.update(visible=num_images >= 4),
        ]

    # Trigger visibility update when the slider value changes
    num_images_slider.change(
        fn=update_visible_images,
        inputs=num_images_slider,
        outputs=[condition_image_2, condition_image_3, condition_image_4],
    )

    run_btn.click(
        fn=run_inference,
        inputs=[
            camera_mode,
            condition_image_1,
            condition_image_2,
            condition_image_3,
            condition_image_4,
            input_camera_pose_format,
            model_camera_pose_format,
            rotation_x,
            rotation_y,
            rotation_z,
            translation_x,
            translation_y,
            translation_z,
            trajectory_extension_factor,
            cfg_scale,
            extra_cfg_scale,
            sample_steps,
            num_images_slider,
            spherical_angle_x,
            spherical_angle_y,
            spherical_radius,
            random_seed,
        ],
        outputs=output_video,
    )

demo.launch()
