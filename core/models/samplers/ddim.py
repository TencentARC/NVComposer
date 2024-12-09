"""SAMPLING ONLY."""

import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm

from core.common import noise_like
from core.models.utils_diffusion import (
    make_ddim_sampling_parameters,
    make_ddim_time_steps,
    rescale_noise_cfg,
)


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_time_steps = model.num_time_steps
        self.schedule = schedule
        self.counter = 0

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(
        self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0.0, verbose=True
    ):
        self.ddim_time_steps = make_ddim_time_steps(
            ddim_discr_method=ddim_discretize,
            num_ddim_time_steps=ddim_num_steps,
            num_ddpm_time_steps=self.ddpm_num_time_steps,
            verbose=verbose,
        )
        alphas_cumprod = self.model.alphas_cumprod
        assert (
            alphas_cumprod.shape[0] == self.ddpm_num_time_steps
        ), "alphas have to be defined for each timestep"

        def to_torch(x):
            return x.clone().detach().to(torch.float32).to(self.model.device)

        if self.model.use_dynamic_rescale:
            self.ddim_scale_arr = self.model.scale_arr[self.ddim_time_steps]
            self.ddim_scale_arr_prev = torch.cat(
                [self.ddim_scale_arr[0:1], self.ddim_scale_arr[:-1]]
            )

        self.register_buffer("betas", to_torch(self.model.betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer(
            "alphas_cumprod_prev", to_torch(self.model.alphas_cumprod_prev)
        )

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            "sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod.cpu()))
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            to_torch(np.sqrt(1.0 - alphas_cumprod.cpu())),
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod.cpu()))
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod.cpu()))
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod",
            to_torch(np.sqrt(1.0 / alphas_cumprod.cpu() - 1)),
        )

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums=alphas_cumprod.cpu(),
            ddim_time_steps=self.ddim_time_steps,
            eta=ddim_eta,
            verbose=verbose,
        )
        self.register_buffer("ddim_sigmas", ddim_sigmas)
        self.register_buffer("ddim_alphas", ddim_alphas)
        self.register_buffer("ddim_alphas_prev", ddim_alphas_prev)
        self.register_buffer("ddim_sqrt_one_minus_alphas", np.sqrt(1.0 - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev)
            / (1 - self.alphas_cumprod)
            * (1 - self.alphas_cumprod / self.alphas_cumprod_prev)
        )
        self.register_buffer(
            "ddim_sigmas_for_original_num_steps", sigmas_for_original_sampling_steps
        )

    @torch.no_grad()
    def sample(
        self,
        S,
        batch_size,
        shape,
        conditioning=None,
        callback=None,
        img_callback=None,
        quantize_x0=False,
        eta=0.0,
        mask=None,
        x0=None,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        verbose=True,
        schedule_verbose=False,
        x_T=None,
        log_every_t=100,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        unconditional_guidance_scale_extra=1.0,
        unconditional_conditioning_extra=None,
        with_extra_returned_data=False,
        **kwargs,
    ):

        # check condition bs
        if conditioning is not None:
            if isinstance(conditioning, dict):
                try:
                    cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                except:
                    cbs = conditioning[list(conditioning.keys())[0]][0].shape[0]

                if cbs != batch_size:
                    print(
                        f"Warning: Got {cbs} conditionings but batch-size is {batch_size}"
                    )
            else:
                if conditioning.shape[0] != batch_size:
                    print(
                        f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}"
                    )

        self.skip_step = self.ddpm_num_time_steps // S
        discr_method = (
            "uniform_trailing" if self.model.rescale_betas_zero_snr else "uniform"
        )
        self.make_schedule(
            ddim_num_steps=S,
            ddim_discretize=discr_method,
            ddim_eta=eta,
            verbose=schedule_verbose,
        )

        # make shape
        if len(shape) == 3:
            C, H, W = shape
            size = (batch_size, C, H, W)
        elif len(shape) == 4:
            T, C, H, W = shape
            size = (batch_size, T, C, H, W)
        else:
            assert False, f"Invalid shape: {shape}."
        out = self.ddim_sampling(
            conditioning,
            size,
            callback=callback,
            img_callback=img_callback,
            quantize_denoised=quantize_x0,
            mask=mask,
            x0=x0,
            ddim_use_original_steps=False,
            noise_dropout=noise_dropout,
            temperature=temperature,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs,
            x_T=x_T,
            log_every_t=log_every_t,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
            unconditional_guidance_scale_extra=unconditional_guidance_scale_extra,
            unconditional_conditioning_extra=unconditional_conditioning_extra,
            verbose=verbose,
            with_extra_returned_data=with_extra_returned_data,
            **kwargs,
        )
        if with_extra_returned_data:
            samples, intermediates, extra_returned_data = out
            return samples, intermediates, extra_returned_data
        else:
            samples, intermediates = out
            return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(
        self,
        cond,
        shape,
        x_T=None,
        ddim_use_original_steps=False,
        callback=None,
        time_steps=None,
        quantize_denoised=False,
        mask=None,
        x0=None,
        img_callback=None,
        log_every_t=100,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        unconditional_guidance_scale_extra=1.0,
        unconditional_conditioning_extra=None,
        verbose=True,
        with_extra_returned_data=False,
        **kwargs,
    ):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device, dtype=self.model.dtype)
            if self.model.bd_noise:
                noise_decor = self.model.bd(img)
                noise_decor = (noise_decor - noise_decor.mean()) / (
                    noise_decor.std() + 1e-5
                )
                noise_f = noise_decor[:, :, 0:1, :, :]
                noise = (
                    np.sqrt(self.model.bd_ratio) * noise_decor[:, :, 1:]
                    + np.sqrt(1 - self.model.bd_ratio) * noise_f
                )
                img = torch.cat([noise_f, noise], dim=2)
        else:
            img = x_T

        if time_steps is None:
            time_steps = (
                self.ddpm_num_time_steps
                if ddim_use_original_steps
                else self.ddim_time_steps
            )
        elif time_steps is not None and not ddim_use_original_steps:
            subset_end = (
                int(
                    min(time_steps / self.ddim_time_steps.shape[0], 1)
                    * self.ddim_time_steps.shape[0]
                )
                - 1
            )
            time_steps = self.ddim_time_steps[:subset_end]

        intermediates = {"x_inter": [img], "pred_x0": [img]}
        time_range = (
            reversed(range(0, time_steps))
            if ddim_use_original_steps
            else np.flip(time_steps)
        )
        total_steps = time_steps if ddim_use_original_steps else time_steps.shape[0]
        if verbose:
            iterator = tqdm(time_range, desc="DDIM Sampler", total=total_steps)
        else:
            iterator = time_range
        # Sampling Loop
        for i, step in enumerate(iterator):
            print(f"Sample: i={i}, step={step}.")
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            print("ts=", ts)
            # use mask to blend noised original latent (img_orig) & new sampled latent (img)
            if mask is not None:
                assert x0 is not None
                img_orig = x0
                # keep original & modify use img
                img = img_orig * mask + (1.0 - mask) * img
            outs = self.p_sample_ddim(
                img,
                cond,
                ts,
                index=index,
                use_original_steps=ddim_use_original_steps,
                quantize_denoised=quantize_denoised,
                temperature=temperature,
                noise_dropout=noise_dropout,
                score_corrector=score_corrector,
                corrector_kwargs=corrector_kwargs,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
                unconditional_guidance_scale_extra=unconditional_guidance_scale_extra,
                unconditional_conditioning_extra=unconditional_conditioning_extra,
                with_extra_returned_data=with_extra_returned_data,
                **kwargs,
            )
            if with_extra_returned_data:
                img, pred_x0, extra_returned_data = outs
            else:
                img, pred_x0 = outs
            if callback:
                callback(i)
            if img_callback:
                img_callback(pred_x0, i)
            # log_every_t = 1
            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates["x_inter"].append(img)
                intermediates["pred_x0"].append(pred_x0)
                # intermediates['extra_returned_data'].append(extra_returned_data)
        if with_extra_returned_data:
            return img, intermediates, extra_returned_data
        return img, intermediates

    def batch_time_transpose(
        self, batch_time_tensor, num_target_views, num_condition_views
    ):
        # Input: N*N; N = T+C
        assert num_target_views + num_condition_views == batch_time_tensor.shape[1]
        target_tensor = batch_time_tensor[:, :num_target_views, ...]  # T*T
        condition_tensor = batch_time_tensor[:, num_target_views:, ...]  # N*C
        target_tensor = target_tensor.transpose(0, 1)  # T*T
        return torch.concat([target_tensor, condition_tensor], dim=1)

    def ddim_batch_shard_step(
        self,
        pred_x0_post_process_function,
        pred_x0_post_process_function_kwargs,
        cond,
        corrector_kwargs,
        ddim_use_original_steps,
        device,
        img,
        index,
        kwargs,
        noise_dropout,
        quantize_denoised,
        score_corrector,
        step,
        temperature,
        with_extra_returned_data,
    ):
        img_list = []
        pred_x0_list = []
        shard_step = 5
        shard_start = 0
        while shard_start < img.shape[0]:
            shard_end = shard_start + shard_step
            if shard_start >= img.shape[0]:
                break
            if shard_end > img.shape[0]:
                shard_end = img.shape[0]
            print(
                f"Sampling Batch Shard: From #{shard_start} to #{shard_end}. Total: {img.shape[0]}."
            )
            sub_img = img[shard_start:shard_end]
            sub_cond = {
                "combined_condition": cond["combined_condition"][shard_start:shard_end],
                "c_crossattn": [
                    cond["c_crossattn"][0][0:1].expand(shard_end - shard_start, -1, -1)
                ],
            }
            ts = torch.full((sub_img.shape[0],), step, device=device, dtype=torch.long)

            _img, _pred_x0 = self.p_sample_ddim(
                sub_img,
                sub_cond,
                ts,
                index=index,
                use_original_steps=ddim_use_original_steps,
                quantize_denoised=quantize_denoised,
                temperature=temperature,
                noise_dropout=noise_dropout,
                score_corrector=score_corrector,
                corrector_kwargs=corrector_kwargs,
                unconditional_guidance_scale=1.0,
                unconditional_conditioning=None,
                unconditional_guidance_scale_extra=1.0,
                unconditional_conditioning_extra=None,
                pred_x0_post_process_function=pred_x0_post_process_function,
                pred_x0_post_process_function_kwargs=pred_x0_post_process_function_kwargs,
                with_extra_returned_data=with_extra_returned_data,
                **kwargs,
            )
            img_list.append(_img)
            pred_x0_list.append(_pred_x0)
            shard_start += shard_step
        img = torch.concat(img_list, dim=0)
        pred_x0 = torch.concat(pred_x0_list, dim=0)
        return img, pred_x0

    @torch.no_grad()
    def p_sample_ddim(
        self,
        x,
        c,
        t,
        index,
        repeat_noise=False,
        use_original_steps=False,
        quantize_denoised=False,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        unconditional_guidance_scale_extra=1.0,
        unconditional_conditioning_extra=None,
        with_extra_returned_data=False,
        **kwargs,
    ):
        b, *_, device = *x.shape, x.device
        if x.dim() == 5:
            is_video = True
        else:
            is_video = False

        extra_returned_data = None
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.0:
            e_t_cfg = self.model.apply_model(x, t, c, **kwargs)  # unet denoiser
            if isinstance(e_t_cfg, tuple):
                e_t_cfg = e_t_cfg[0]
                extra_returned_data = e_t_cfg[1:]
        else:
            # with unconditional condition
            if isinstance(c, torch.Tensor) or isinstance(c, dict):
                e_t = self.model.apply_model(x, t, c, **kwargs)
                e_t_uncond = self.model.apply_model(
                    x, t, unconditional_conditioning, **kwargs
                )
                if (
                    unconditional_guidance_scale_extra != 1.0
                    and unconditional_conditioning_extra is not None
                ):
                    print(f"Using extra CFG: {unconditional_guidance_scale_extra}...")
                    e_t_uncond_extra = self.model.apply_model(
                        x, t, unconditional_conditioning_extra, **kwargs
                    )
                else:
                    e_t_uncond_extra = None
            else:
                raise NotImplementedError

            if isinstance(e_t, tuple):
                e_t = e_t[0]
                extra_returned_data = e_t[1:]

            if isinstance(e_t_uncond, tuple):
                e_t_uncond = e_t_uncond[0]
            if isinstance(e_t_uncond_extra, tuple):
                e_t_uncond_extra = e_t_uncond_extra[0]

            # text cfg
            if (
                unconditional_guidance_scale_extra != 1.0
                and unconditional_conditioning_extra is not None
            ):
                e_t_cfg = (
                    e_t_uncond
                    + unconditional_guidance_scale * (e_t - e_t_uncond)
                    + unconditional_guidance_scale_extra * (e_t - e_t_uncond_extra)
                )
            else:
                e_t_cfg = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

            if self.model.rescale_betas_zero_snr:
                e_t_cfg = rescale_noise_cfg(e_t_cfg, e_t, guidance_rescale=0.7)

        if self.model.parameterization == "v":
            e_t = self.model.predict_eps_from_z_and_v(x, t, e_t_cfg)
        else:
            e_t = e_t_cfg

        if score_corrector is not None:
            assert self.model.parameterization == "eps", "not implemented"
            e_t = score_corrector.modify_score(
                self.model, e_t, x, t, c, **corrector_kwargs
            )

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = (
            self.model.alphas_cumprod_prev
            if use_original_steps
            else self.ddim_alphas_prev
        )
        sqrt_one_minus_alphas = (
            self.model.sqrt_one_minus_alphas_cumprod
            if use_original_steps
            else self.ddim_sqrt_one_minus_alphas
        )
        sigmas = (
            self.model.ddim_sigmas_for_original_num_steps
            if use_original_steps
            else self.ddim_sigmas
        )
        # select parameters corresponding to the currently considered timestep

        if is_video:
            size = (b, 1, 1, 1, 1)
        else:
            size = (b, 1, 1, 1)
        a_t = torch.full(size, alphas[index], device=device)
        a_prev = torch.full(size, alphas_prev[index], device=device)
        sigma_t = torch.full(size, sigmas[index], device=device)
        sqrt_one_minus_at = torch.full(
            size, sqrt_one_minus_alphas[index], device=device
        )

        # current prediction for x_0
        if self.model.parameterization != "v":
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        else:
            pred_x0 = self.model.predict_start_from_z_and_v(x, t, e_t_cfg)

        if self.model.use_dynamic_rescale:
            scale_t = torch.full(size, self.ddim_scale_arr[index], device=device)
            prev_scale_t = torch.full(
                size, self.ddim_scale_arr_prev[index], device=device
            )
            rescale = prev_scale_t / scale_t
            pred_x0 *= rescale

        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        dir_xt = (1.0 - a_prev - sigma_t**2).sqrt() * e_t

        noise = noise_like(x.shape, device, repeat_noise)
        if self.model.bd_noise:
            noise_decor = self.model.bd(noise)
            noise_decor = (noise_decor - noise_decor.mean()) / (
                noise_decor.std() + 1e-5
            )
            noise_f = noise_decor[:, :, 0:1, :, :]
            noise = (
                np.sqrt(self.model.bd_ratio) * noise_decor[:, :, 1:]
                + np.sqrt(1 - self.model.bd_ratio) * noise_f
            )
            noise = torch.cat([noise_f, noise], dim=2)
        noise = sigma_t * noise * temperature

        if noise_dropout > 0.0:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)

        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        if with_extra_returned_data:
            return x_prev, pred_x0, extra_returned_data
        return x_prev, pred_x0
