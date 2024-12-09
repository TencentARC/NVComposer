from core.models.samplers.ddim import DDIMSampler
import glob
import json
import os
import sys
from collections import OrderedDict

import numpy as np
import torch
import torchvision
from PIL import Image

sys.path.insert(1, os.path.join(sys.path[0], "..", ".."))


def batch_ddim_sampling(
    model,
    cond,
    noise_shape,
    n_samples=1,
    ddim_steps=50,
    ddim_eta=1.0,
    cfg_scale=1.0,
    temporal_cfg_scale=None,
    use_cat_ucg=False,
    **kwargs,
):
    ddim_sampler = DDIMSampler(model)
    uncond_type = model.uncond_type
    batch_size = noise_shape[0]

    # construct unconditional guidance
    if cfg_scale != 1.0:
        if uncond_type == "empty_seq":
            prompts = batch_size * [""]
            # prompts = N * T * [""]  # if is_image_batch=True
            uc_emb = model.get_learned_conditioning(prompts)
        elif uncond_type == "zero_embed":
            c_emb = cond["c_crossattn"][0] if isinstance(cond, dict) else cond
            uc_emb = torch.zeros_like(c_emb)

        # process image condition
        if hasattr(model, "embedder"):
            uc_img = torch.zeros(noise_shape[0], 3, 224, 224).to(model.device)
            # img: b c h w >> b l c
            uc_img = model.get_image_embeds(uc_img)
            uc_emb = torch.cat([uc_emb, uc_img], dim=1)

        if isinstance(cond, dict):
            uc = {key: cond[key] for key in cond.keys()}
            uc.update({"c_crossattn": [uc_emb]})
            # special CFG for frame concatenation
            if use_cat_ucg and hasattr(model, "cond_concat") and model.cond_concat:
                uc_cat = torch.zeros(
                    noise_shape[0], model.cond_channels, *noise_shape[2:]
                ).to(model.device)
                uc.update({"c_concat": [uc_cat]})
        else:
            uc = [uc_emb]
    else:
        uc = None
    # uc.update({'fps': torch.tensor([-4]*batch_size).to(model.device).long()})
    # sampling
    noise = torch.randn(noise_shape, device=model.device)
    # x_T = repeat(noise[:,:,:1,:,:], 'b c l h w -> b c (l t) h w', t=noise_shape[2])
    # x_T = 0.2 * x_T + 0.8 * torch.randn(noise_shape, device=model.device)
    x_T = None
    batch_variants = []
    # batch_variants1, batch_variants2 = [], []
    for _ in range(n_samples):
        if ddim_sampler is not None:
            samples, _ = ddim_sampler.sample(
                S=ddim_steps,
                conditioning=cond,
                batch_size=noise_shape[0],
                shape=noise_shape[1:],
                verbose=False,
                unconditional_guidance_scale=cfg_scale,
                unconditional_conditioning=uc,
                eta=ddim_eta,
                temporal_length=noise_shape[2],
                conditional_guidance_scale_temporal=temporal_cfg_scale,
                x_T=x_T,
                **kwargs,
            )
        # reconstruct from latent to pixel space
        batch_images = model.decode_first_stage(samples)
        batch_variants.append(batch_images)
        """
        pred_x0_list, x_iter_list = _['pred_x0'], _['x_inter']
        steps = [0, 15, 25, 30, 35, 40, 43, 46, 49, 50]
        for nn in steps:
            pred_x0 = pred_x0_list[nn]
            x_iter = x_iter_list[nn]
            batch_images_x0 = model.decode_first_stage(pred_x0)
            batch_variants1.append(batch_images_x0)
            batch_images_xt = model.decode_first_stage(x_iter)
            batch_variants2.append(batch_images_xt)
        """
    # batch, <samples>, c, t, h, w
    batch_variants = torch.stack(batch_variants, dim=1)
    # batch_variants1 = torch.stack(batch_variants1, dim=1)
    # batch_variants2 = torch.stack(batch_variants2, dim=1)
    # return batch_variants1, batch_variants2
    return batch_variants


def batch_sliding_interpolation(
    model,
    cond,
    base_videos,
    base_stride,
    noise_shape,
    n_samples=1,
    ddim_steps=50,
    ddim_eta=1.0,
    cfg_scale=1.0,
    temporal_cfg_scale=None,
    **kwargs,
):
    """
    Current implementation has a flaw: the inter-episode keyframe is used as pre-last and cur-first, so keyframe repeated.
    For example, cond_frames=[0,4,7], model.temporal_length=8, base_stride=4, then
    base frame  : 0   4   8   12  16  20  24  28
    interplation: (0~7)   (8~15)  (16~23) (20~27)
    """
    b, c, t, h, w = noise_shape
    base_z0 = model.encode_first_stage(base_videos)
    unit_length = model.temporal_length
    n_base_frames = base_videos.shape[2]
    n_refs = len(model.cond_frames)
    sliding_steps = (n_base_frames - 1) // (n_refs - 1)
    sliding_steps = (
        sliding_steps + 1 if (n_base_frames - 1) % (n_refs - 1) > 0 else sliding_steps
    )

    cond_mask = model.cond_mask.to("cuda")
    proxy_z0 = torch.zeros((b, c, unit_length, h, w), dtype=torch.float32).to("cuda")
    batch_samples = None
    last_offset = None
    for idx in range(sliding_steps):
        base_idx = idx * (n_refs - 1)
        # check index overflow
        if base_idx + n_refs > n_base_frames:
            last_offset = base_idx - (n_base_frames - n_refs)
            base_idx = n_base_frames - n_refs
        cond_z0 = base_z0[:, :, base_idx : base_idx + n_refs, :, :]
        proxy_z0[:, :, model.cond_frames, :, :] = cond_z0

        if "c_concat" in cond:
            c_cat, text_emb = cond["c_concat"][0], cond["c_crossattn"][0]
            episode_idx = idx * unit_length
            if last_offset is not None:
                episode_idx = episode_idx - last_offset * base_stride
            cond_idx = {
                "c_concat": [
                    c_cat[:, :, episode_idx : episode_idx + unit_length, :, :]
                ],
                "c_crossattn": [text_emb],
            }
        else:
            cond_idx = cond
        noise_shape_idx = [b, c, unit_length, h, w]
        # batch, <samples>, c, t, h, w
        batch_idx = batch_ddim_sampling(
            model,
            cond_idx,
            noise_shape_idx,
            n_samples,
            ddim_steps,
            ddim_eta,
            cfg_scale,
            temporal_cfg_scale,
            mask=cond_mask,
            x0=proxy_z0,
            **kwargs,
        )

        if batch_samples is None:
            batch_samples = batch_idx
        else:
            # b,s,c,t,h,w
            if last_offset is None:
                batch_samples = torch.cat(
                    [batch_samples[:, :, :, :-1, :, :], batch_idx], dim=3
                )
            else:
                batch_samples = torch.cat(
                    [
                        batch_samples[:, :, :, :-1, :, :],
                        batch_idx[:, :, :, last_offset * base_stride :, :, :],
                    ],
                    dim=3,
                )

    return batch_samples


def get_filelist(data_dir, ext="*"):
    file_list = glob.glob(os.path.join(data_dir, "*.%s" % ext))
    file_list.sort()
    return file_list


def get_dirlist(path):
    list = []
    if os.path.exists(path):
        files = os.listdir(path)
        for file in files:
            m = os.path.join(path, file)
            if os.path.isdir(m):
                list.append(m)
    list.sort()
    return list


def load_model_checkpoint(model, ckpt, adapter_ckpt=None):
    def load_checkpoint(model, ckpt, full_strict):
        state_dict = torch.load(ckpt, map_location="cpu", weights_only=True)
        try:
            # deepspeed
            new_pl_sd = OrderedDict()
            for key in state_dict["module"].keys():
                new_pl_sd[key[16:]] = state_dict["module"][key]
            model.load_state_dict(new_pl_sd, strict=full_strict)
        except:
            if "state_dict" in list(state_dict.keys()):
                state_dict = state_dict["state_dict"]
            model.load_state_dict(state_dict, strict=full_strict)
        return model

    if adapter_ckpt:
        # main model
        load_checkpoint(model, ckpt, full_strict=False)
        print(">>> model checkpoint loaded.")
        # adapter
        state_dict = torch.load(adapter_ckpt, map_location="cpu")
        if "state_dict" in list(state_dict.keys()):
            state_dict = state_dict["state_dict"]
        model.adapter.load_state_dict(state_dict, strict=True)
        print(">>> adapter checkpoint loaded.")
    else:
        load_checkpoint(model, ckpt, full_strict=False)
        print(">>> model checkpoint loaded.")
    return model


def load_prompts(prompt_file):
    f = open(prompt_file, "r")
    prompt_list = []
    for idx, line in enumerate(f.readlines()):
        l = line.strip()
        if len(l) != 0:
            prompt_list.append(l)
        f.close()
    return prompt_list


def load_camera_poses(filepath_list, video_frames=16):
    pose_list = []
    for filepath in filepath_list:
        with open(filepath, "r") as f:
            pose = json.load(f)
        pose = np.array(pose)  # [t, 12]
        pose = torch.tensor(pose).float()  # [t, 12]
        assert (
            pose.shape[0] == video_frames
        ), f"conditional pose frames Not matching the target frames [{video_frames}]."
        pose_list.append(pose)
    batch_poses = torch.stack(pose_list, dim=0)
    # shape [b,t,12,1]
    return batch_poses[..., None]


def save_videos(
    batch_tensors: torch.Tensor, save_dir: str, filenames: list[str], fps: int = 10
):
    # b,samples,t,c,h,w
    n_samples = batch_tensors.shape[1]
    for idx, vid_tensor in enumerate(batch_tensors):
        video = vid_tensor.detach().cpu()
        video = torch.clamp(video.float(), -1.0, 1.0)
        video = video.permute(1, 0, 2, 3, 4)  # t,n,c,h,w
        frame_grids = [
            torchvision.utils.make_grid(framesheet, nrow=int(n_samples))
            for framesheet in video
        ]  # [3, 1*h, n*w]
        # stack in temporal dim [t, 3, n*h, w]
        grid = torch.stack(frame_grids, dim=0)
        grid = (grid + 1.0) / 2.0
        grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)
        savepath = os.path.join(save_dir, f"{filenames[idx]}.mp4")
        torchvision.io.write_video(
            savepath, grid, fps=fps, video_codec="h264", options={"crf": "10"}
        )
