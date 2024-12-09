import logging
from collections import OrderedDict
from contextlib import contextmanager
from functools import partial

import numpy as np
from einops import rearrange
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision.utils import make_grid
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

from core.modules.networks.unet_modules import TASK_IDX_IMAGE, TASK_IDX_RAY
from utils.utils import instantiate_from_config
from core.ema import LitEma
from core.distributions import DiagonalGaussianDistribution
from core.models.utils_diffusion import make_beta_schedule, rescale_zero_terminal_snr
from core.models.samplers.ddim import DDIMSampler
from core.basics import disabled_train
from core.common import extract_into_tensor, noise_like, exists, default

main_logger = logging.getLogger("main_logger")


class BD(nn.Module):
    def __init__(self, G=10):
        super(BD, self).__init__()

        self.momentum = 0.9
        self.register_buffer("running_wm", torch.eye(G).expand(G, G))
        self.running_wm = None

    def forward(self, x, T=5, eps=1e-5):
        N, C, G, H, W = x.size()
        x = torch.permute(x, [0, 2, 1, 3, 4])
        x_in = x.transpose(0, 1).contiguous().view(G, -1)
        if self.training:
            mean = x_in.mean(-1, keepdim=True)
            xc = x_in - mean
            d, m = x_in.size()
            P = [None] * (T + 1)
            P[0] = torch.eye(G, device=x.device)
            Sigma = (torch.matmul(xc, xc.transpose(0, 1))) / float(m) + P[0] * eps
            rTr = (Sigma * P[0]).sum([0, 1], keepdim=True).reciprocal()
            Sigma_N = Sigma * rTr
            wm = torch.linalg.solve_triangular(
                torch.linalg.cholesky(Sigma_N), P[0], upper=False
            )
            self.running_wm = self.momentum * self.running_wm + (1 - self.momentum) * wm
        else:
            wm = self.running_wm

        x_out = wm @ x_in
        x_out = x_out.view(G, N, C, H, W).permute([1, 2, 0, 3, 4]).contiguous()

        return x_out


class AbstractDDPM(pl.LightningModule):

    def __init__(
        self,
        unet_config,
        time_steps=1000,
        beta_schedule="linear",
        loss_type="l2",
        monitor=None,
        use_ema=True,
        first_stage_key="image",
        image_size=256,
        channels=3,
        log_every_t=100,
        clip_denoised=True,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
        given_betas=None,
        original_elbo_weight=0.0,
        # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
        v_posterior=0.0,
        l_simple_weight=1.0,
        conditioning_key=None,
        parameterization="eps",
        rescale_betas_zero_snr=False,
        scheduler_config=None,
        use_positional_encodings=False,
        learn_logvar=False,
        logvar_init=0.0,
        bd_noise=False,
    ):
        super().__init__()
        assert parameterization in [
            "eps",
            "x0",
            "v",
        ], 'currently only supporting "eps" and "x0" and "v"'
        self.parameterization = parameterization
        main_logger.info(
            f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode"
        )
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.channels = channels
        self.cond_channels = unet_config.params.in_channels - channels
        self.temporal_length = unet_config.params.temporal_length
        self.image_size = image_size
        self.bd_noise = bd_noise

        if self.bd_noise:
            self.bd = BD(G=self.temporal_length)

        if isinstance(self.image_size, int):
            self.image_size = [self.image_size, self.image_size]
        self.use_positional_encodings = use_positional_encodings
        self.model = DiffusionWrapper(unet_config)
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            main_logger.info(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.rescale_betas_zero_snr = rescale_betas_zero_snr
        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        self.linear_end = None
        self.linear_start = None
        self.num_time_steps: int = 1000

        if monitor is not None:
            self.monitor = monitor

        self.register_schedule(
            given_betas=given_betas,
            beta_schedule=beta_schedule,
            time_steps=time_steps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )

        self.given_betas = given_betas
        self.beta_schedule = beta_schedule
        self.time_steps = time_steps
        self.cosine_s = cosine_s

        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_time_steps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
            * noise
        )

    def predict_start_from_z_and_v(self, x_t, t, v):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def predict_eps_from_z_and_v(self, x_t, t, v):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * v
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
            * x_t
        )

    def get_v(self, x, noise, t):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape) * noise
            - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * x
        )

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                main_logger.info(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    main_logger.info(f"{context}: Restored training weights")

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == "l1":
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == "l2":
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction="none")
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    def _get_rows_from_list(self, samples):
        n_imgs_per_row = len(samples)
        denoise_grid = rearrange(samples, "n b c h w -> b n c h w")
        denoise_grid = rearrange(denoise_grid, "b n c h w -> (b n) c h w")
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid


class DualStreamMultiViewDiffusionModel(AbstractDDPM):

    def __init__(
        self,
        first_stage_config,
        data_key_images,
        data_key_rays,
        data_key_text_condition=None,
        ckpt_path=None,
        cond_stage_config=None,
        num_time_steps_cond=None,
        cond_stage_trainable=False,
        cond_stage_forward=None,
        conditioning_key=None,
        uncond_prob=0.2,
        uncond_type="empty_seq",
        scale_factor=1.0,
        scale_by_std=False,
        use_noise_offset=False,
        use_dynamic_rescale=False,
        base_scale=0.3,
        turning_step=400,
        per_frame_auto_encoding=False,
        # added for LVDM
        encoder_type="2d",
        cond_frames=None,
        logdir=None,
        empty_params_only=False,
        # Image Condition
        cond_img_config=None,
        image_proj_model_config=None,
        random_cond=False,
        padding=False,
        cond_concat=False,
        frame_mask=False,
        use_camera_pose_query_transformer=False,
        with_cond_binary_mask=False,
        apply_condition_mask_in_training_loss=True,
        separate_noise_and_condition=False,
        condition_padding_with_anchor=False,
        ray_as_image=False,
        use_task_embedding=False,
        use_ray_decoder_loss_high_frequency_isolation=False,
        disable_ray_stream=False,
        ray_loss_weight=1.0,
        train_with_multi_view_feature_alignment=False,
        use_text_cross_attention_condition=True,
        *args,
        **kwargs,
    ):

        self.image_proj_model = None
        self.apply_condition_mask_in_training_loss = (
            apply_condition_mask_in_training_loss
        )
        self.separate_noise_and_condition = separate_noise_and_condition
        self.condition_padding_with_anchor = condition_padding_with_anchor
        self.use_text_cross_attention_condition = use_text_cross_attention_condition

        self.data_key_images = data_key_images
        self.data_key_rays = data_key_rays
        self.data_key_text_condition = data_key_text_condition

        self.num_time_steps_cond = default(num_time_steps_cond, 1)
        self.scale_by_std = scale_by_std
        assert self.num_time_steps_cond <= kwargs["time_steps"]
        self.shorten_cond_schedule = self.num_time_steps_cond > 1
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)

        self.cond_stage_trainable = cond_stage_trainable
        self.empty_params_only = empty_params_only
        self.per_frame_auto_encoding = per_frame_auto_encoding
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer("scale_factor", torch.tensor(scale_factor))
        self.use_noise_offset = use_noise_offset
        self.use_dynamic_rescale = use_dynamic_rescale
        if use_dynamic_rescale:
            scale_arr1 = np.linspace(1.0, base_scale, turning_step)
            scale_arr2 = np.full(self.num_time_steps, base_scale)
            scale_arr = np.concatenate((scale_arr1, scale_arr2))
            to_torch = partial(torch.tensor, dtype=torch.float32)
            self.register_buffer("scale_arr", to_torch(scale_arr))
        self.instantiate_first_stage(first_stage_config)

        if self.use_text_cross_attention_condition and cond_stage_config is not None:
            self.instantiate_cond_stage(cond_stage_config)

        self.first_stage_config = first_stage_config
        self.cond_stage_config = cond_stage_config
        self.clip_denoised = False

        self.cond_stage_forward = cond_stage_forward
        self.encoder_type = encoder_type
        assert encoder_type in ["2d", "3d"]
        self.uncond_prob = uncond_prob
        self.classifier_free_guidance = True if uncond_prob > 0 else False
        assert uncond_type in ["zero_embed", "empty_seq"]
        self.uncond_type = uncond_type

        if cond_frames is not None:
            frame_len = self.temporal_length
            assert cond_frames[-1] < frame_len, main_logger.info(
                f"Error: conditioning frame index must not be greater than {frame_len}!"
            )
            cond_mask = torch.zeros(frame_len, dtype=torch.float32)
            cond_mask[cond_frames] = 1.0
            self.cond_mask = cond_mask[None, None, :, None, None]
        else:
            self.cond_mask = None

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)
            self.restarted_from_ckpt = True

        self.logdir = logdir
        self.with_cond_binary_mask = with_cond_binary_mask
        self.random_cond = random_cond
        self.padding = padding
        self.cond_concat = cond_concat
        self.frame_mask = frame_mask
        self.use_img_context = True if cond_img_config is not None else False
        self.use_camera_pose_query_transformer = use_camera_pose_query_transformer
        if self.use_img_context:
            self.init_img_embedder(cond_img_config, freeze=True)
            self.init_projector(image_proj_model_config, trainable=True)

        self.ray_as_image = ray_as_image
        self.use_task_embedding = use_task_embedding
        self.use_ray_decoder_loss_high_frequency_isolation = (
            use_ray_decoder_loss_high_frequency_isolation
        )
        self.disable_ray_stream = disable_ray_stream
        if disable_ray_stream:
            assert (
                not ray_as_image
                and not self.model.diffusion_model.use_ray_decoder
                and not self.model.diffusion_model.use_ray_decoder_residual
            ), "Options related to ray decoder should not be enabled when disabling ray stream."
            assert (
                not use_task_embedding
                and not self.model.diffusion_model.use_task_embedding
            ), "Task embedding should not be enabled when disabling ray stream."
            assert (
                not self.model.diffusion_model.use_addition_ray_output_head
            ), "Additional ray output head should not be enabled when disabling ray stream."
            assert (
                not self.model.diffusion_model.use_lora_for_rays_in_output_blocks
            ), "LoRA for rays should not be enabled when disabling ray stream."
        self.ray_loss_weight = ray_loss_weight
        self.train_with_multi_view_feature_alignment = False
        if train_with_multi_view_feature_alignment:
            print(f"MultiViewFeatureExtractor is ignored during inference.")

    def init_from_ckpt(self, checkpoint_path):
        main_logger.info(f"Initializing model from checkpoint {checkpoint_path}...")

        def grab_ipa_weight(state_dict):
            ipa_state_dict = OrderedDict()
            for n in list(state_dict.keys()):
                if "to_k_ip" in n or "to_v_ip" in n:
                    ipa_state_dict[n] = state_dict[n]
                elif "image_proj_model" in n:
                    if (
                        self.use_camera_pose_query_transformer
                        and "image_proj_model.latents" in n
                    ):
                        ipa_state_dict[n] = torch.cat(
                            [state_dict[n] for i in range(16)], dim=1
                        )
                    else:
                        ipa_state_dict[n] = state_dict[n]
            return ipa_state_dict

        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if "module" in state_dict.keys():
            # deepspeed
            target_state_dict = OrderedDict()
            for key in state_dict["module"].keys():
                target_state_dict[key[16:]] = state_dict["module"][key]
        elif "state_dict" in list(state_dict.keys()):
            target_state_dict = state_dict["state_dict"]
        else:
            raise KeyError("Weight key is not found in the state dict.")
        ipa_state_dict = grab_ipa_weight(target_state_dict)
        self.load_state_dict(ipa_state_dict, strict=False)
        main_logger.info("Checkpoint loaded.")

    def init_img_embedder(self, config, freeze=True):
        embedder = instantiate_from_config(config)
        if freeze:
            self.embedder = embedder.eval()
            self.embedder.train = disabled_train
            for param in self.embedder.parameters():
                param.requires_grad = False

    def make_cond_schedule(
        self,
    ):
        self.cond_ids = torch.full(
            size=(self.num_time_steps,),
            fill_value=self.num_time_steps - 1,
            dtype=torch.long,
        )
        ids = torch.round(
            torch.linspace(0, self.num_time_steps - 1, self.num_time_steps_cond)
        ).long()
        self.cond_ids[: self.num_time_steps_cond] = ids

    def init_projector(self, config, trainable):
        self.image_proj_model = instantiate_from_config(config)
        if not trainable:
            self.image_proj_model.eval()
            self.image_proj_model.train = disabled_train
            for param in self.image_proj_model.parameters():
                param.requires_grad = False

    @staticmethod
    def pad_cond_images(batch_images):
        h, w = batch_images.shape[-2:]
        border = (w - h) // 2
        # use padding at (W_t,W_b,H_t,H_b)
        batch_images = torch.nn.functional.pad(
            batch_images, (0, 0, border, border), "constant", 0
        )
        return batch_images

    # Never delete this func: it is used in log_images() and inference stage
    def get_image_embeds(self, batch_images, batch=None):
        # input shape: b c h w
        if self.padding:
            batch_images = self.pad_cond_images(batch_images)
        img_token = self.embedder(batch_images)
        if self.use_camera_pose_query_transformer:
            batch_size, num_views, _ = batch["target_poses"].shape
            img_emb = self.image_proj_model(
                img_token, batch["target_poses"].reshape(batch_size, num_views, 12)
            )
        else:
            img_emb = self.image_proj_model(img_token)

        return img_emb

    @staticmethod
    def get_input(batch, k):
        x = batch[k]
        """
        # for image batch from image loader
        if len(x.shape) == 4:
            x = rearrange(x, 'b h w c -> b c h w')
        """
        x = x.to(memory_format=torch.contiguous_format)  # .float()
        return x

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx=None):
        # only for very first batch, reset the self.scale_factor
        if (
            self.scale_by_std
            and self.current_epoch == 0
            and self.global_step == 0
            and batch_idx == 0
            and not self.restarted_from_ckpt
        ):
            assert (
                self.scale_factor == 1.0
            ), "rather not use custom rescaling and std-rescaling simultaneously"
            # set rescale weight to 1./std of encodings
            main_logger.info("## USING STD-RESCALING ###")
            x = self.get_input(batch, self.first_stage_key)
            x = x.to(self.device)
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            del self.scale_factor
            self.register_buffer("scale_factor", 1.0 / z.flatten().std())
            main_logger.info(f"setting self.scale_factor to {self.scale_factor}")
            main_logger.info("## USING STD-RESCALING ###")
            main_logger.info(f"std={z.flatten().std()}")

    def register_schedule(
        self,
        given_betas=None,
        beta_schedule="linear",
        time_steps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
    ):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(
                beta_schedule,
                time_steps,
                linear_start=linear_start,
                linear_end=linear_end,
                cosine_s=cosine_s,
            )

        if self.rescale_betas_zero_snr:
            betas = rescale_zero_terminal_snr(betas)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        (time_steps,) = betas.shape
        self.num_time_steps = int(time_steps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert (
            alphas_cumprod.shape[0] == self.num_time_steps
        ), "alphas have to be defined for each timestep"

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod",
            to_torch(np.sqrt(1.0 / (alphas_cumprod + 1e-5))),
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod",
            to_torch(np.sqrt(1.0 / (alphas_cumprod + 1e-5) - 1)),
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (
            1.0 - alphas_cumprod_prev
        ) / (1.0 - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            to_torch(
                (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
            ),
        )

        if self.parameterization == "eps":
            lvlb_weights = self.betas**2 / (
                2
                * self.posterior_variance
                * to_torch(alphas)
                * (1 - self.alphas_cumprod)
            )
        elif self.parameterization == "x0":
            lvlb_weights = (
                0.5
                * np.sqrt(torch.Tensor(alphas_cumprod))
                / (2.0 * 1 - torch.Tensor(alphas_cumprod))
            )
        elif self.parameterization == "v":
            lvlb_weights = torch.ones_like(
                self.betas**2
                / (
                    2
                    * self.posterior_variance
                    * to_torch(alphas)
                    * (1 - self.alphas_cumprod)
                )
            )
        else:
            raise NotImplementedError("mu not supported")
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer("lvlb_weights", lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            model = instantiate_from_config(config)
            self.cond_stage_model = model.eval()
            self.cond_stage_model.train = disabled_train
            for param in self.cond_stage_model.parameters():
                param.requires_grad = False
        else:
            model = instantiate_from_config(config)
            self.cond_stage_model = model

    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, "encode") and callable(
                self.cond_stage_model.encode
            ):
                c = self.cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

    def get_first_stage_encoding(self, encoder_posterior, noise=None):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample(noise=noise)
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(
                f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented"
            )
        return self.scale_factor * z

    @torch.no_grad()
    def encode_first_stage(self, x):
        assert x.dim() == 5 or x.dim() == 4, (
            "Images should be a either 5-dimensional (batched image sequence) "
            "or 4-dimensional (batched images)."
        )
        if (
            self.encoder_type == "2d"
            and x.dim() == 5
            and not self.per_frame_auto_encoding
        ):
            b, t, _, _, _ = x.shape
            x = rearrange(x, "b t c h w -> (b t) c h w")
            reshape_back = True
        else:
            b, _, _, _, _ = x.shape
            t = 1
            reshape_back = False

        if not self.per_frame_auto_encoding:
            encoder_posterior = self.first_stage_model.encode(x)
            results = self.get_first_stage_encoding(encoder_posterior).detach()
        else:
            results = []
            for index in range(x.shape[1]):
                frame_batch = self.first_stage_model.encode(x[:, index, :, :, :])
                frame_result = self.get_first_stage_encoding(frame_batch).detach()
                results.append(frame_result)
            results = torch.stack(results, dim=1)

        if reshape_back:
            results = rearrange(results, "(b t) c h w -> b t c h w", b=b, t=t)

        return results

    def decode_core(self, z, **kwargs):
        assert z.dim() == 5 or z.dim() == 4, (
            "Latents should be a either 5-dimensional (batched latent sequence) "
            "or 4-dimensional (batched latents)."
        )

        if (
            self.encoder_type == "2d"
            and z.dim() == 5
            and not self.per_frame_auto_encoding
        ):
            b, t, _, _, _ = z.shape
            z = rearrange(z, "b t c h w -> (b t) c h w")
            reshape_back = True
        else:
            b, _, _, _, _ = z.shape
            t = 1
            reshape_back = False

        if not self.per_frame_auto_encoding:
            z = 1.0 / self.scale_factor * z
            results = self.first_stage_model.decode(z, **kwargs)
        else:
            results = []
            for index in range(z.shape[1]):
                frame_z = 1.0 / self.scale_factor * z[:, index, :, :, :]
                frame_result = self.first_stage_model.decode(frame_z, **kwargs)
                results.append(frame_result)
            results = torch.stack(results, dim=1)

        if reshape_back:
            results = rearrange(results, "(b t) c h w -> b t c h w", b=b, t=t)
        return results

    @torch.no_grad()
    def decode_first_stage(self, z, **kwargs):
        return self.decode_core(z, **kwargs)

    def differentiable_decode_first_stage(self, z, **kwargs):
        return self.decode_core(z, **kwargs)

    def get_batch_input(
        self,
        batch,
        random_drop_training_conditions,
        return_reconstructed_target_images=False,
    ):
        combined_images = batch[self.data_key_images]
        clean_combined_image_latents = self.encode_first_stage(combined_images)
        mask_preserving_target = batch["mask_preserving_target"].reshape(
            batch["mask_preserving_target"].size(0),
            batch["mask_preserving_target"].size(1),
            1,
            1,
            1,
        )
        mask_preserving_condition = 1.0 - mask_preserving_target
        if self.ray_as_image:
            clean_combined_ray_images = batch[self.data_key_rays]
            clean_combined_ray_o_latents = self.encode_first_stage(
                clean_combined_ray_images[:, :, :3, :, :]
            )
            clean_combined_ray_d_latents = self.encode_first_stage(
                clean_combined_ray_images[:, :, 3:, :, :]
            )
            clean_combined_rays = torch.concat(
                [clean_combined_ray_o_latents, clean_combined_ray_d_latents], dim=2
            )

            if self.condition_padding_with_anchor:
                condition_ray_images = batch["condition_rays"]
                condition_ray_o_images = self.encode_first_stage(
                    condition_ray_images[:, :, :3, :, :]
                )
                condition_ray_d_images = self.encode_first_stage(
                    condition_ray_images[:, :, 3:, :, :]
                )
                condition_rays = torch.concat(
                    [condition_ray_o_images, condition_ray_d_images], dim=2
                )
            else:
                condition_rays = clean_combined_rays * mask_preserving_target
        else:
            clean_combined_rays = batch[self.data_key_rays]

            if self.condition_padding_with_anchor:
                condition_rays = batch["condition_rays"]
            else:
                condition_rays = clean_combined_rays * mask_preserving_target

        if self.condition_padding_with_anchor:
            condition_images_latents = self.encode_first_stage(
                batch["condition_images"]
            )
        else:
            condition_images_latents = (
                clean_combined_image_latents * mask_preserving_condition
            )

        if random_drop_training_conditions:
            random_num = torch.rand(
                combined_images.size(0), device=combined_images.device
            )
        else:
            random_num = torch.ones(
                combined_images.size(0), device=combined_images.device
            )

        text_feature_condition_mask = rearrange(
            random_num < 2 * self.uncond_prob, "n -> n 1 1"
        )
        image_feature_condition_mask = 1 - rearrange(
            (random_num >= self.uncond_prob).float()
            * (random_num < 3 * self.uncond_prob).float(),
            "n -> n 1 1 1 1",
        )
        ray_condition_mask = 1 - rearrange(
            (random_num >= 1.5 * self.uncond_prob).float()
            * (random_num < 3.5 * self.uncond_prob).float(),
            "n -> n 1 1 1 1",
        )
        mask_preserving_first_target = batch[
            "mask_only_preserving_first_target"
        ].reshape(
            batch["mask_only_preserving_first_target"].size(0),
            batch["mask_only_preserving_first_target"].size(1),
            1,
            1,
            1,
        )
        mask_preserving_first_condition = batch[
            "mask_only_preserving_first_condition"
        ].reshape(
            batch["mask_only_preserving_first_condition"].size(0),
            batch["mask_only_preserving_first_condition"].size(1),
            1,
            1,
            1,
        )
        mask_preserving_anchors = (
            mask_preserving_first_target + mask_preserving_first_condition
        )
        mask_randomly_preserving_first_target = torch.where(
            ray_condition_mask.repeat(1, mask_preserving_first_target.size(1), 1, 1, 1)
            == 1.0,
            1.0,
            mask_preserving_first_target,
        )
        mask_randomly_preserving_first_condition = torch.where(
            image_feature_condition_mask.repeat(
                1, mask_preserving_first_condition.size(1), 1, 1, 1
            )
            == 1.0,
            1.0,
            mask_preserving_first_condition,
        )

        if self.use_text_cross_attention_condition:
            text_cond_key = self.data_key_text_condition
            text_cond = batch[text_cond_key]
            if isinstance(text_cond, dict) or isinstance(text_cond, list):
                full_text_cond_emb = self.get_learned_conditioning(text_cond)
            else:
                full_text_cond_emb = self.get_learned_conditioning(
                    text_cond.to(self.device)
                )
            null_text_cond_emb = self.get_learned_conditioning([""])
            text_cond_emb = torch.where(
                text_feature_condition_mask,
                null_text_cond_emb,
                full_text_cond_emb.detach(),
            )

        batch_size, num_views, _, _, _ = batch[self.data_key_images].shape
        if self.condition_padding_with_anchor:
            condition_images = batch["condition_images"]
        else:
            condition_images = combined_images * mask_preserving_condition
        if random_drop_training_conditions:
            condition_image_for_embedder = rearrange(
                condition_images * image_feature_condition_mask,
                "b t c h w -> (b t) c h w",
            )
        else:
            condition_image_for_embedder = rearrange(
                condition_images, "b t c h w -> (b t) c h w"
            )
        img_token = self.embedder(condition_image_for_embedder)
        if self.use_camera_pose_query_transformer:
            img_emb = self.image_proj_model(
                img_token, batch["target_poses"].reshape(batch_size, num_views, 12)
            )
        else:
            img_emb = self.image_proj_model(img_token)

        img_emb = rearrange(
            img_emb, "(b t) s d -> b (t s) d", b=batch_size, t=num_views
        )
        if self.use_text_cross_attention_condition:
            c_crossattn = [torch.cat([text_cond_emb, img_emb], dim=1)]
        else:
            c_crossattn = [img_emb]

        cond_dict = {
            "c_crossattn": c_crossattn,
            "target_camera_poses": batch["target_and_condition_camera_poses"]
            * batch["mask_preserving_target"].unsqueeze(-1),
        }

        if self.disable_ray_stream:
            clean_gt = torch.cat([clean_combined_image_latents], dim=2)
        else:
            clean_gt = torch.cat(
                [clean_combined_image_latents, clean_combined_rays], dim=2
            )
        if random_drop_training_conditions:
            combined_condition = torch.cat(
                [
                    condition_images_latents * mask_randomly_preserving_first_condition,
                    condition_rays * mask_randomly_preserving_first_target,
                ],
                dim=2,
            )
        else:
            combined_condition = torch.cat(
                [condition_images_latents, condition_rays], dim=2
            )

        uncond_combined_condition = torch.cat(
            [
                condition_images_latents * mask_preserving_anchors,
                condition_rays * mask_preserving_anchors,
            ],
            dim=2,
        )

        mask_full_for_input = torch.cat(
            [
                mask_preserving_condition.repeat(
                    1, 1, condition_images_latents.size(2), 1, 1
                ),
                mask_preserving_target.repeat(1, 1, condition_rays.size(2), 1, 1),
            ],
            dim=2,
        )
        cond_dict.update(
            {
                "mask_preserving_target": mask_preserving_target,
                "mask_preserving_condition": mask_preserving_condition,
                "combined_condition": combined_condition,
                "uncond_combined_condition": uncond_combined_condition,
                "clean_combined_rays": clean_combined_rays,
                "mask_full_for_input": mask_full_for_input,
                "num_cond_images": rearrange(
                    batch["num_cond_images"].float(), "b -> b 1 1 1 1"
                ),
                "num_target_images": rearrange(
                    batch["num_target_images"].float(), "b -> b 1 1 1 1"
                ),
            }
        )

        out = [clean_gt, cond_dict]
        if return_reconstructed_target_images:
            target_images_reconstructed = self.decode_first_stage(
                clean_combined_image_latents
            )
            out.append(target_images_reconstructed)
        return out

    def get_dynamic_scales(self, t, spin_step=400):
        base_scale = self.base_scale
        scale_t = torch.where(
            t < spin_step,
            t * (base_scale - 1.0) / spin_step + 1.0,
            base_scale * torch.ones_like(t),
        )
        return scale_t

    def forward(self, x, c, **kwargs):
        t = torch.randint(
            0, self.num_time_steps, (x.shape[0],), device=self.device
        ).long()
        if self.use_dynamic_rescale:
            x = x * extract_into_tensor(self.scale_arr, t, x.shape)
        return self.p_losses(x, c, t, **kwargs)

    def extract_feature(self, batch, t, **kwargs):
        z, cond = self.get_batch_input(
            batch,
            random_drop_training_conditions=False,
            return_reconstructed_target_images=False,
        )
        if self.use_dynamic_rescale:
            z = z * extract_into_tensor(self.scale_arr, t, z.shape)
        noise = torch.randn_like(z)
        if self.use_noise_offset:
            noise = noise + 0.1 * torch.randn(
                noise.shape[0], noise.shape[1], 1, 1, 1
            ).to(self.device)
        x_noisy = self.q_sample(x_start=z, t=t, noise=noise)
        x_noisy = self.process_x_with_condition(x_noisy, condition_dict=cond)
        c_crossattn = torch.cat(cond["c_crossattn"], 1)
        target_camera_poses = cond["target_camera_poses"]
        x_pred, features = self.model(
            x_noisy,
            t,
            context=c_crossattn,
            return_output_block_features=True,
            camera_poses=target_camera_poses,
            **kwargs,
        )
        return x_pred, features, z

    def apply_model(self, x_noisy, t, cond, features_to_return=None, **kwargs):
        if not isinstance(cond, dict):
            if not isinstance(cond, list):
                cond = [cond]
            key = (
                "c_concat" if self.model.conditioning_key == "concat" else "c_crossattn"
            )
            cond = {key: cond}

        c_crossattn = torch.cat(cond["c_crossattn"], 1)
        x_noisy = self.process_x_with_condition(x_noisy, condition_dict=cond)
        target_camera_poses = cond["target_camera_poses"]
        if self.use_task_embedding:
            x_pred_images = self.model(
                x_noisy,
                t,
                context=c_crossattn,
                task_idx=TASK_IDX_IMAGE,
                camera_poses=target_camera_poses,
                **kwargs,
            )
            x_pred_rays = self.model(
                x_noisy,
                t,
                context=c_crossattn,
                task_idx=TASK_IDX_RAY,
                camera_poses=target_camera_poses,
                **kwargs,
            )
            x_pred = torch.concat([x_pred_images, x_pred_rays], dim=2)
        elif features_to_return is not None:
            x_pred, features = self.model(
                x_noisy,
                t,
                context=c_crossattn,
                return_input_block_features="input" in features_to_return,
                return_middle_feature="middle" in features_to_return,
                return_output_block_features="output" in features_to_return,
                camera_poses=target_camera_poses,
                **kwargs,
            )
            return x_pred, features
        elif self.train_with_multi_view_feature_alignment:
            x_pred, aligned_features = self.model(
                x_noisy,
                t,
                context=c_crossattn,
                camera_poses=target_camera_poses,
                **kwargs,
            )
            return x_pred, aligned_features
        else:
            x_pred = self.model(
                x_noisy,
                t,
                context=c_crossattn,
                camera_poses=target_camera_poses,
                **kwargs,
            )
        return x_pred

    def process_x_with_condition(self, x_noisy, condition_dict):
        combined_condition = condition_dict["combined_condition"]
        if self.separate_noise_and_condition:
            if self.disable_ray_stream:
                x_noisy = torch.concat([x_noisy, combined_condition], dim=2)
            else:
                x_noisy = torch.concat(
                    [
                        x_noisy[:, :, :4, :, :],
                        combined_condition[:, :, :4, :, :],
                        x_noisy[:, :, 4:, :, :],
                        combined_condition[:, :, 4:, :, :],
                    ],
                    dim=2,
                )
        else:
            assert (
                not self.use_ray_decoder_regression
            ), "`separate_noise_and_condition` must be True when enabling `use_ray_decoder_regression`."
            mask_preserving_target = condition_dict["mask_preserving_target"]
            mask_preserving_condition = condition_dict["mask_preserving_condition"]
            mask_for_combined_condition = torch.cat(
                [
                    mask_preserving_target.repeat(1, 1, 4, 1, 1),
                    mask_preserving_condition.repeat(1, 1, 6, 1, 1),
                ]
            )
            mask_for_x_noisy = torch.cat(
                [
                    mask_preserving_target.repeat(1, 1, 4, 1, 1),
                    mask_preserving_condition.repeat(1, 1, 6, 1, 1),
                ]
            )
            x_noisy = (
                x_noisy * mask_for_x_noisy
                + combined_condition * mask_for_combined_condition
            )

        return x_noisy

    def p_losses(self, x_start, cond, t, noise=None, **kwargs):

        noise = default(noise, lambda: torch.randn_like(x_start))

        if self.use_noise_offset:
            noise = noise + 0.1 * torch.randn(
                noise.shape[0], noise.shape[1], 1, 1, 1
            ).to(self.device)

        # noise em !!!
        if self.bd_noise:
            noise_decor = self.bd(noise)
            noise_decor = (noise_decor - noise_decor.mean()) / (
                noise_decor.std() + 1e-5
            )
            noise_f = noise_decor[:, :, 0:1, :, :]
            noise = (
                np.sqrt(self.bd_ratio) * noise_decor[:, :, 1:]
                + np.sqrt(1 - self.bd_ratio) * noise_f
            )
            noise = torch.cat([noise_f, noise], dim=2)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        if self.train_with_multi_view_feature_alignment:
            model_output, aligned_features = self.apply_model(
                x_noisy, t, cond, **kwargs
            )

            aligned_middle_feature = rearrange(
                aligned_features,
                "(b t) c h w -> b (t c h w)",
                b=cond["pts_anchor_to_all"].size(0),
                t=cond["pts_anchor_to_all"].size(1),
            )
            target_multi_view_feature = rearrange(
                torch.concat(
                    [cond["pts_anchor_to_all"], cond["pts_all_to_anchor"]], dim=2
                ),
                "b t c h w -> b (t c h w)",
            ).to(aligned_middle_feature.device)
        else:
            model_output = self.apply_model(x_noisy, t, cond, **kwargs)

        loss_dict = {}
        prefix = "train" if self.training else "val"

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        if self.apply_condition_mask_in_training_loss:
            mask_full_for_output = 1.0 - cond["mask_full_for_input"]
            model_output = model_output * mask_full_for_output
            target = target * mask_full_for_output
        loss_simple = self.get_loss(model_output, target, mean=False)
        if self.ray_loss_weight != 1.0:
            loss_simple[:, :, 4:, :, :] = (
                loss_simple[:, :, 4:, :, :] * self.ray_loss_weight
            )
        if self.apply_condition_mask_in_training_loss:
            # Ray loss: predicted items = # of condition images
            num_total_images = cond["num_cond_images"] + cond["num_target_images"]
            weight_for_image_loss = num_total_images / cond["num_target_images"]
            weight_for_ray_loss = num_total_images / cond["num_cond_images"]
            loss_simple[:, :, :4, :, :] = (
                loss_simple[:, :, :4, :, :] * weight_for_image_loss
            )
            # Ray loss: predicted items = # of condition images
            loss_simple[:, :, 4:, :, :] = (
                loss_simple[:, :, 4:, :, :] * weight_for_ray_loss
            )

        loss_dict.update({f"{prefix}/loss_images": loss_simple[:, :, 0:4, :, :].mean()})
        if not self.disable_ray_stream:
            loss_dict.update(
                {f"{prefix}/loss_rays": loss_simple[:, :, 4:, :, :].mean()}
            )
        loss_simple = loss_simple.mean([1, 2, 3, 4])
        loss_dict.update({f"{prefix}/loss_simple": loss_simple.mean()})

        if self.logvar.device is not self.device:
            self.logvar = self.logvar.to(self.device)
        logvar_t = self.logvar[t]
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        if self.learn_logvar:
            loss_dict.update({f"{prefix}/loss_gamma": loss.mean()})
            loss_dict.update({"logvar": self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        if self.train_with_multi_view_feature_alignment:
            multi_view_feature_alignment_loss = 0.25 * torch.nn.functional.mse_loss(
                aligned_middle_feature, target_multi_view_feature
            )
            loss += multi_view_feature_alignment_loss
            loss_dict.update(
                {f"{prefix}/loss_mv_feat_align": multi_view_feature_alignment_loss}
            )

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(
            dim=(1, 2, 3, 4)
        )
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f"{prefix}/loss_vlb": loss_vlb})
        loss += self.original_elbo_weight * loss_vlb
        loss_dict.update({f"{prefix}/loss": loss})

        return loss, loss_dict

    def _get_denoise_row_from_list(self, samples, desc=""):
        denoise_row = []
        for zd in tqdm(samples, desc=desc):
            denoise_row.append(self.decode_first_stage(zd.to(self.device)))
        n_log_time_steps = len(denoise_row)

        denoise_row = torch.stack(denoise_row)  # n_log_time_steps, b, C, H, W

        if denoise_row.dim() == 5:
            denoise_grid = rearrange(denoise_row, "n b c h w -> b n c h w")
            denoise_grid = rearrange(denoise_grid, "b n c h w -> (b n) c h w")
            denoise_grid = make_grid(denoise_grid, nrow=n_log_time_steps)
        elif denoise_row.dim() == 6:
            video_length = denoise_row.shape[3]
            denoise_grid = rearrange(denoise_row, "n b c t h w -> b n c t h w")
            denoise_grid = rearrange(denoise_grid, "b n c t h w -> (b n) c t h w")
            denoise_grid = rearrange(denoise_grid, "n c t h w -> (n t) c h w")
            denoise_grid = make_grid(denoise_grid, nrow=video_length)
        else:
            raise ValueError

        return denoise_grid

    @torch.no_grad()
    def log_images(
        self,
        batch,
        sample=True,
        ddim_steps=50,
        ddim_eta=1.0,
        plot_denoise_rows=False,
        unconditional_guidance_scale=1.0,
        **kwargs,
    ):
        """log images for LatentDiffusion"""
        use_ddim = ddim_steps is not None
        log = dict()
        z, cond, x_rec = self.get_batch_input(
            batch,
            random_drop_training_conditions=False,
            return_reconstructed_target_images=True,
        )
        b, t, c, h, w = x_rec.shape
        log["num_cond_images_str"] = batch["num_cond_images_str"]
        log["caption"] = batch["caption"]
        if "condition_images" in batch:
            log["input_condition_images_all"] = batch["condition_images"]
        log["input_condition_image_latents_masked"] = cond["combined_condition"][
            :, :, 0:3, :, :
        ]
        log["input_condition_rays_o_masked"] = (
            cond["combined_condition"][:, :, 4:7, :, :] / 5.0
        )
        log["input_condition_rays_d_masked"] = (
            cond["combined_condition"][:, :, 7:, :, :] / 5.0
        )
        log["gt_images_after_vae"] = x_rec
        if self.train_with_multi_view_feature_alignment:
            log["pts_anchor_to_all"] = cond["pts_anchor_to_all"]
            log["pts_all_to_anchor"] = cond["pts_all_to_anchor"]
            log["pts_anchor_to_all"] = (
                log["pts_anchor_to_all"] - torch.min(log["pts_anchor_to_all"])
            ) / torch.max(log["pts_anchor_to_all"])
            log["pts_all_to_anchor"] = (
                log["pts_all_to_anchor"] - torch.min(log["pts_all_to_anchor"])
            ) / torch.max(log["pts_all_to_anchor"])

        if self.ray_as_image:
            log["gt_rays_o"] = batch["combined_rays"][:, :, 0:3, :, :]
            log["gt_rays_d"] = batch["combined_rays"][:, :, 3:, :, :]
        else:
            log["gt_rays_o"] = batch["combined_rays"][:, :, 0:3, :, :] / 5.0
            log["gt_rays_d"] = batch["combined_rays"][:, :, 3:, :, :] / 5.0

        if sample:
            # get uncond embedding for classifier-free guidance sampling
            if unconditional_guidance_scale != 1.0:
                uc = self.get_unconditional_dict_for_sampling(batch, cond, x_rec)
            else:
                uc = None

            with self.ema_scope("Plotting"):
                out = self.sample_log(
                    cond=cond,
                    batch_size=b,
                    ddim=use_ddim,
                    ddim_steps=ddim_steps,
                    eta=ddim_eta,
                    unconditional_guidance_scale=unconditional_guidance_scale,
                    unconditional_conditioning=uc,
                    mask=self.cond_mask,
                    x0=z,
                    with_extra_returned_data=False,
                    **kwargs,
                )
                samples, z_denoise_row = out
            per_instance_decoding = False

            if per_instance_decoding:
                x_sample_images = []
                for idx in range(b):
                    sample_image = samples[idx : idx + 1, :, 0:4, :, :]
                    x_sample_image = self.decode_first_stage(sample_image)
                    x_sample_images.append(x_sample_image)
                x_sample_images = torch.cat(x_sample_images, dim=0)
            else:
                x_sample_images = self.decode_first_stage(samples[:, :, 0:4, :, :])
            log["sample_images"] = x_sample_images

            if not self.disable_ray_stream:
                if self.ray_as_image:
                    log["sample_rays_o"] = self.decode_first_stage(
                        samples[:, :, 4:8, :, :]
                    )
                    log["sample_rays_d"] = self.decode_first_stage(
                        samples[:, :, 8:, :, :]
                    )
                else:
                    log["sample_rays_o"] = samples[:, :, 4:7, :, :] / 5.0
                    log["sample_rays_d"] = samples[:, :, 7:, :, :] / 5.0

            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        return log

    def get_unconditional_dict_for_sampling(self, batch, cond, x_rec, is_extra=False):
        b, t, c, h, w = x_rec.shape
        if self.use_text_cross_attention_condition:
            if self.uncond_type == "empty_seq":
                # NVComposer's cross attention layers accept multi-view images
                prompts = b * [""]
                # prompts = b * t * [""]  # if is_image_batch=True
                uc_emb = self.get_learned_conditioning(prompts)
            elif self.uncond_type == "zero_embed":
                c_emb = cond["c_crossattn"][0] if isinstance(cond, dict) else cond
                uc_emb = torch.zeros_like(c_emb)
        else:
            uc_emb = None

        # process image condition
        if not is_extra:
            if hasattr(self, "embedder"):
                # uc_img = torch.zeros_like(x[:, :, 0, ...])  # b c h w
                uc_img = torch.zeros(
                    # b c h w
                    size=(b * t, c, h, w),
                    dtype=x_rec.dtype,
                    device=x_rec.device,
                )
                # img: b c h w >> b l c
                uc_img = self.get_image_embeds(uc_img, batch)

                # Modified: The uc embeddings should be reshaped for valid post-processing
                uc_img = rearrange(
                    uc_img, "(b t) s d -> b (t s) d", b=b, t=uc_img.shape[0] // b
                )
                if uc_emb is None:
                    uc_emb = uc_img
                else:
                    uc_emb = torch.cat([uc_emb, uc_img], dim=1)
            uc = {key: cond[key] for key in cond.keys()}
            uc.update({"c_crossattn": [uc_emb]})
        else:
            uc = {key: cond[key] for key in cond.keys()}
            uc.update({"combined_condition": uc["uncond_combined_condition"]})

        return uc

    def p_mean_variance(
        self,
        x,
        c,
        t,
        clip_denoised: bool,
        return_x0=False,
        score_corrector=None,
        corrector_kwargs=None,
        **kwargs,
    ):
        t_in = t
        model_out = self.apply_model(x, t_in, c, **kwargs)

        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(
                self, model_out, x, t, c, **corrector_kwargs
            )

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )

        if return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(
        self,
        x,
        c,
        t,
        clip_denoised=False,
        repeat_noise=False,
        return_x0=False,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        **kwargs,
    ):
        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(
            x=x,
            c=c,
            t=t,
            clip_denoised=clip_denoised,
            return_x0=return_x0,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs,
            **kwargs,
        )
        if return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.0:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)

        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        if return_x0:
            return (
                model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise,
                x0,
            )
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(
        self,
        cond,
        shape,
        return_intermediates=False,
        x_T=None,
        verbose=True,
        callback=None,
        time_steps=None,
        mask=None,
        x0=None,
        img_callback=None,
        start_T=None,
        log_every_t=None,
        **kwargs,
    ):
        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = [img]
        if time_steps is None:
            time_steps = self.num_time_steps
        if start_T is not None:
            time_steps = min(time_steps, start_T)

        iterator = (
            tqdm(reversed(range(0, time_steps)), desc="Sampling t", total=time_steps)
            if verbose
            else reversed(range(0, time_steps))
        )

        if mask is not None:
            assert x0 is not None
            # spatial size has to match
            assert x0.shape[2:3] == mask.shape[2:3]

        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != "hybrid"
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img = self.p_sample(
                img, cond, ts, clip_denoised=self.clip_denoised, **kwargs
            )

            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1.0 - mask) * img

            if i % log_every_t == 0 or i == time_steps - 1:
                intermediates.append(img)
            if callback:
                callback(i)
            if img_callback:
                img_callback(img, i)

        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(
        self,
        cond,
        batch_size=16,
        return_intermediates=False,
        x_T=None,
        verbose=True,
        time_steps=None,
        mask=None,
        x0=None,
        shape=None,
        **kwargs,
    ):
        if shape is None:
            shape = (batch_size, self.channels, self.temporal_length, *self.image_size)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {
                    key: (
                        cond[key][:batch_size]
                        if not isinstance(cond[key], list)
                        else list(map(lambda x: x[:batch_size], cond[key]))
                    )
                    for key in cond
                }
            else:
                cond = (
                    [c[:batch_size] for c in cond]
                    if isinstance(cond, list)
                    else cond[:batch_size]
                )
        return self.p_sample_loop(
            cond,
            shape,
            return_intermediates=return_intermediates,
            x_T=x_T,
            verbose=verbose,
            time_steps=time_steps,
            mask=mask,
            x0=x0,
            **kwargs,
        )

    @torch.no_grad()
    def sample_log(
        self,
        cond,
        batch_size,
        ddim,
        ddim_steps,
        with_extra_returned_data=False,
        **kwargs,
    ):
        if ddim:
            ddim_sampler = DDIMSampler(self)
            shape = (self.temporal_length, self.channels, *self.image_size)
            out = ddim_sampler.sample(
                ddim_steps,
                batch_size,
                shape,
                cond,
                verbose=True,
                with_extra_returned_data=with_extra_returned_data,
                **kwargs,
            )
            if with_extra_returned_data:
                samples, intermediates, extra_returned_data = out
                return samples, intermediates, extra_returned_data
            else:
                samples, intermediates = out
                return samples, intermediates

        else:
            samples, intermediates = self.sample(
                cond=cond, batch_size=batch_size, return_intermediates=True, **kwargs
            )

            return samples, intermediates


class DiffusionWrapper(pl.LightningModule):
    def __init__(self, diff_model_config):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)

    def forward(self, x, c, **kwargs):
        return self.diffusion_model(x, c, **kwargs)
