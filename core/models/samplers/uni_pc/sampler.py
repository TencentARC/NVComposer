"""SAMPLING ONLY."""

import torch

from .uni_pc import NoiseScheduleVP, model_wrapper, UniPC


class UniPCSampler(object):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model

        def to_torch(x):
            return x.clone().detach().to(torch.float32).to(model.device)

        self.register_buffer("alphas_cumprod", to_torch(model.alphas_cumprod))

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    @torch.no_grad()
    def sample(
        self,
        S,
        batch_size,
        shape,
        conditioning=None,
        x_T=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
    ):
        # sampling
        T, C, H, W = shape
        size = (batch_size, T, C, H, W)

        device = self.model.betas.device
        if x_T is None:
            img = torch.randn(size, device=device)
        else:
            img = x_T

        ns = NoiseScheduleVP("discrete", alphas_cumprod=self.alphas_cumprod)

        model_fn = model_wrapper(
            lambda x, t, c: self.model.apply_model(x, t, c),
            ns,
            model_type="v",
            guidance_type="classifier-free",
            condition=conditioning,
            unconditional_condition=unconditional_conditioning,
            guidance_scale=unconditional_guidance_scale,
        )

        uni_pc = UniPC(model_fn, ns, predict_x0=True, thresholding=False)
        x = uni_pc.sample(
            img,
            steps=S,
            skip_type="time_uniform",
            method="multistep",
            order=2,
            lower_order_final=True,
        )

        return x.to(device), None
