from functools import partial
from abc import abstractmethod
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from core.models.utils_diffusion import timestep_embedding
from core.common import gradient_checkpoint
from core.basics import zero_module, conv_nd, linear, avg_pool_nd, normalization
from core.modules.attention import SpatialTransformer, TemporalTransformer

TASK_IDX_IMAGE = 0
TASK_IDX_RAY = 1


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(
        self, x, emb, context=None, batch_size=None, with_lora=False, time_steps=None
    ):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb, batch_size=batch_size)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context, with_lora=with_lora)
            elif isinstance(layer, TemporalTransformer):
                x = rearrange(x, "(b f) c h w -> b c f h w", b=batch_size)
                x = layer(x, context, with_lora=with_lora, time_steps=time_steps)
                x = rearrange(x, "b c f h w -> (b f) c h w")
            else:
                x = layer(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims,
                self.channels,
                self.out_channels,
                3,
                stride=stride,
                padding=padding,
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(
                dims, self.channels, self.out_channels, 3, padding=padding
            )

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    :param use_temporal_conv: if True, use the temporal convolution.
    :param use_image_dataset: if True, the temporal parameters will not be optimized.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        use_conv=False,
        up=False,
        down=False,
        use_temporal_conv=False,
        tempspatial_aware=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.use_temporal_conv = use_temporal_conv

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

        if self.use_temporal_conv:
            self.temopral_conv = TemporalConvBlock(
                self.out_channels,
                self.out_channels,
                dropout=0.1,
                spatial_aware=tempspatial_aware,
            )

    def forward(self, x, emb, batch_size=None):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        input_tuple = (x, emb)
        if batch_size:
            forward_batchsize = partial(self._forward, batch_size=batch_size)
            return gradient_checkpoint(
                forward_batchsize, input_tuple, self.parameters(), self.use_checkpoint
            )
        return gradient_checkpoint(
            self._forward, input_tuple, self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb, batch_size=None):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        h = self.skip_connection(x) + h

        if self.use_temporal_conv and batch_size:
            h = rearrange(h, "(b t) c h w -> b c t h w", b=batch_size)
            h = self.temopral_conv(h)
            h = rearrange(h, "b c t h w -> (b t) c h w")
        return h


class TemporalConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels=None, dropout=0.0, spatial_aware=False
    ):
        super(TemporalConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        th_kernel_shape = (3, 1, 1) if not spatial_aware else (3, 3, 1)
        th_padding_shape = (1, 0, 0) if not spatial_aware else (1, 1, 0)
        tw_kernel_shape = (3, 1, 1) if not spatial_aware else (3, 1, 3)
        tw_padding_shape = (1, 0, 0) if not spatial_aware else (1, 0, 1)

        # conv layers
        self.conv1 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv3d(
                in_channels, out_channels, th_kernel_shape, padding=th_padding_shape
            ),
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv3d(
                out_channels, in_channels, tw_kernel_shape, padding=tw_padding_shape
            ),
        )
        self.conv3 = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv3d(
                out_channels, in_channels, th_kernel_shape, padding=th_padding_shape
            ),
        )
        self.conv4 = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv3d(
                out_channels, in_channels, tw_kernel_shape, padding=tw_padding_shape
            ),
        )

        # zero out the last layer params,so the conv block is identity
        nn.init.zeros_(self.conv4[-1].weight)
        nn.init.zeros_(self.conv4[-1].bias)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        return identity + x


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: in_channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0.0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        context_dim=None,
        use_scale_shift_norm=False,
        resblock_updown=False,
        num_heads=-1,
        num_head_channels=-1,
        transformer_depth=1,
        use_linear=False,
        use_checkpoint=False,
        temporal_conv=False,
        tempspatial_aware=False,
        temporal_attention=True,
        use_relative_position=True,
        use_causal_attention=False,
        temporal_length=None,
        use_fp16=False,
        addition_attention=False,
        temporal_selfatt_only=True,
        image_cross_attention=False,
        image_cross_attention_scale_learnable=False,
        default_fs=4,
        fs_condition=False,
        use_spatial_temporal_attention=False,
        # >>> Extra Ray Options
        use_addition_ray_output_head=False,
        ray_channels=6,
        use_lora_for_rays_in_output_blocks=False,
        use_task_embedding=False,
        use_ray_decoder=False,
        use_ray_decoder_residual=False,
        full_spatial_temporal_attention=False,
        enhance_multi_view_correspondence=False,
        camera_pose_condition=False,
        use_feature_alignment=False,
    ):
        super(UNetModel, self).__init__()
        if num_heads == -1:
            assert (
                num_head_channels != -1
            ), "Either num_heads or num_head_channels has to be set"
        if num_head_channels == -1:
            assert (
                num_heads != -1
            ), "Either num_heads or num_head_channels has to be set"

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.temporal_attention = temporal_attention
        time_embed_dim = model_channels * 4
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        temporal_self_att_only = True
        self.addition_attention = addition_attention
        self.temporal_length = temporal_length
        self.image_cross_attention = image_cross_attention
        self.image_cross_attention_scale_learnable = (
            image_cross_attention_scale_learnable
        )
        self.default_fs = default_fs
        self.fs_condition = fs_condition
        self.use_spatial_temporal_attention = use_spatial_temporal_attention

        # >>> Extra Ray Options
        self.use_addition_ray_output_head = use_addition_ray_output_head
        self.use_lora_for_rays_in_output_blocks = use_lora_for_rays_in_output_blocks
        if self.use_lora_for_rays_in_output_blocks:
            assert (
                use_addition_ray_output_head
            ), "`use_addition_ray_output_head` is required to be True when using LoRA for rays in output blocks."
            assert (
                not use_task_embedding
            ), "`use_task_embedding` cannot be True when `use_lora_for_rays_in_output_blocks` is enabled."
        if self.use_addition_ray_output_head:
            print("Using additional ray output head...")
            assert (self.out_channels == 4) or (
                4 + ray_channels == self.out_channels
            ), f"`out_channels`={out_channels} is invalid."
            self.out_channels = 4
            out_channels = 4
            self.ray_channels = ray_channels
        self.use_ray_decoder = use_ray_decoder
        if use_ray_decoder:
            assert (
                not use_task_embedding
            ), "`use_task_embedding` cannot be True when `use_ray_decoder_layers` is enabled."
            assert (
                use_addition_ray_output_head
            ), "`use_addition_ray_output_head` must be True when `use_ray_decoder_layers` is enabled."
        self.use_ray_decoder_residual = use_ray_decoder_residual

        # >>> Time/Task Embedding Blocks
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        if fs_condition:
            self.fps_embedding = nn.Sequential(
                linear(model_channels, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )
            nn.init.zeros_(self.fps_embedding[-1].weight)
            nn.init.zeros_(self.fps_embedding[-1].bias)

        if camera_pose_condition:
            self.camera_pose_condition = True
            self.camera_pose_embedding = nn.Sequential(
                linear(12, model_channels),
                nn.SiLU(),
                linear(model_channels, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )
            nn.init.zeros_(self.camera_pose_embedding[-1].weight)
            nn.init.zeros_(self.camera_pose_embedding[-1].bias)

        self.use_task_embedding = use_task_embedding
        if use_task_embedding:
            assert (
                not use_lora_for_rays_in_output_blocks
            ), "`use_lora_for_rays_in_output_blocks` and `use_task_embedding` cannot be True at the same time."
            assert (
                use_addition_ray_output_head
            ), "`use_addition_ray_output_head` is required to be True when `use_task_embedding` is enabled."
            self.task_embedding = nn.Sequential(
                linear(model_channels, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )
            nn.init.zeros_(self.task_embedding[-1].weight)
            nn.init.zeros_(self.task_embedding[-1].bias)
            self.task_parameters = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.zeros(size=[model_channels], requires_grad=True)
                    ),
                    nn.Parameter(
                        torch.zeros(size=[model_channels], requires_grad=True)
                    ),
                ]
            )

        # >>> Input Block
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        if self.addition_attention:
            self.init_attn = TimestepEmbedSequential(
                TemporalTransformer(
                    model_channels,
                    n_heads=8,
                    d_head=num_head_channels,
                    depth=transformer_depth,
                    context_dim=context_dim,
                    use_checkpoint=use_checkpoint,
                    only_self_att=temporal_selfatt_only,
                    causal_attention=False,
                    relative_position=use_relative_position,
                    temporal_length=temporal_length,
                )
            )

        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        tempspatial_aware=tempspatial_aware,
                        use_temporal_conv=temporal_conv,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    layers.append(
                        SpatialTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth,
                            context_dim=context_dim,
                            use_linear=use_linear,
                            use_checkpoint=use_checkpoint,
                            disable_self_attn=False,
                            video_length=temporal_length,
                            image_cross_attention=self.image_cross_attention,
                            image_cross_attention_scale_learnable=self.image_cross_attention_scale_learnable,
                        )
                    )
                    if self.temporal_attention:
                        layers.append(
                            TemporalTransformer(
                                ch,
                                num_heads,
                                dim_head,
                                depth=transformer_depth,
                                context_dim=context_dim,
                                use_linear=use_linear,
                                use_checkpoint=use_checkpoint,
                                only_self_att=temporal_self_att_only,
                                causal_attention=use_causal_attention,
                                relative_position=use_relative_position,
                                temporal_length=temporal_length,
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        layers = [
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                tempspatial_aware=tempspatial_aware,
                use_temporal_conv=temporal_conv,
            ),
            SpatialTransformer(
                ch,
                num_heads,
                dim_head,
                depth=transformer_depth,
                context_dim=context_dim,
                use_linear=use_linear,
                use_checkpoint=use_checkpoint,
                disable_self_attn=False,
                video_length=temporal_length,
                image_cross_attention=self.image_cross_attention,
                image_cross_attention_scale_learnable=self.image_cross_attention_scale_learnable,
            ),
        ]
        if self.temporal_attention:
            layers.append(
                TemporalTransformer(
                    ch,
                    num_heads,
                    dim_head,
                    depth=transformer_depth,
                    context_dim=context_dim,
                    use_linear=use_linear,
                    use_checkpoint=use_checkpoint,
                    only_self_att=temporal_self_att_only,
                    causal_attention=use_causal_attention,
                    relative_position=use_relative_position,
                    temporal_length=temporal_length,
                )
            )
        layers.append(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                tempspatial_aware=tempspatial_aware,
                use_temporal_conv=temporal_conv,
            )
        )

        # >>> Middle Block
        self.middle_block = TimestepEmbedSequential(*layers)

        # >>> Ray Decoder
        if use_ray_decoder:
            self.ray_decoder_blocks = nn.ModuleList([])

        # >>> Output Block
        is_first_layer = True
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        tempspatial_aware=tempspatial_aware,
                        use_temporal_conv=temporal_conv,
                    )
                ]
                if use_ray_decoder:
                    if self.use_ray_decoder_residual:
                        ray_residual_ch = ich
                    else:
                        ray_residual_ch = 0
                    ray_decoder_layers = [
                        ResBlock(
                            (ch if is_first_layer else (ch // 10)) + ray_residual_ch,
                            time_embed_dim,
                            dropout,
                            out_channels=mult * model_channels // 10,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            tempspatial_aware=tempspatial_aware,
                            use_temporal_conv=True,
                        )
                    ]
                    is_first_layer = False
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    layers.append(
                        SpatialTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth,
                            context_dim=context_dim,
                            use_linear=use_linear,
                            use_checkpoint=use_checkpoint,
                            disable_self_attn=False,
                            video_length=temporal_length,
                            image_cross_attention=self.image_cross_attention,
                            image_cross_attention_scale_learnable=self.image_cross_attention_scale_learnable,
                            enable_lora=self.use_lora_for_rays_in_output_blocks,
                        )
                    )
                    if self.temporal_attention:
                        layers.append(
                            TemporalTransformer(
                                ch,
                                num_heads,
                                dim_head,
                                depth=transformer_depth,
                                context_dim=context_dim,
                                use_linear=use_linear,
                                use_checkpoint=use_checkpoint,
                                only_self_att=temporal_self_att_only,
                                causal_attention=use_causal_attention,
                                relative_position=use_relative_position,
                                temporal_length=temporal_length,
                                use_extra_spatial_temporal_self_attention=use_spatial_temporal_attention,
                                enable_lora=self.use_lora_for_rays_in_output_blocks,
                                full_spatial_temporal_attention=full_spatial_temporal_attention,
                                enhance_multi_view_correspondence=enhance_multi_view_correspondence,
                            )
                        )
                if level and i == num_res_blocks:
                    out_ch = ch
                    # out_ray_ch = ray_ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    if use_ray_decoder:
                        ray_decoder_layers.append(
                            ResBlock(
                                ch // 10,
                                time_embed_dim,
                                dropout,
                                out_channels=out_ch // 10,
                                dims=dims,
                                use_checkpoint=use_checkpoint,
                                use_scale_shift_norm=use_scale_shift_norm,
                                up=True,
                            )
                            if resblock_updown
                            else Upsample(
                                ch // 10,
                                conv_resample,
                                dims=dims,
                                out_channels=out_ch // 10,
                            )
                        )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                if use_ray_decoder:
                    self.ray_decoder_blocks.append(
                        TimestepEmbedSequential(*ray_decoder_layers)
                    )

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

        if self.use_addition_ray_output_head:
            ray_model_channels = model_channels // 10
            self.ray_output_head = nn.Sequential(
                normalization(ray_model_channels),
                nn.SiLU(),
                conv_nd(dims, ray_model_channels, ray_model_channels, 3, padding=1),
                nn.SiLU(),
                conv_nd(dims, ray_model_channels, ray_model_channels, 3, padding=1),
                nn.SiLU(),
                zero_module(
                    conv_nd(dims, ray_model_channels, self.ray_channels, 3, padding=1)
                ),
            )
        self.use_feature_alignment = use_feature_alignment
        if self.use_feature_alignment:
            self.feature_alignment_adapter = FeatureAlignmentAdapter(
                time_embed_dim=time_embed_dim, use_checkpoint=use_checkpoint
            )

    def forward(
        self,
        x,
        time_steps,
        context=None,
        features_adapter=None,
        fs=None,
        task_idx=None,
        camera_poses=None,
        return_input_block_features=False,
        return_middle_feature=False,
        return_output_block_features=False,
        **kwargs,
    ):
        intermediate_features = {}
        if return_input_block_features:
            intermediate_features["input"] = []
        if return_output_block_features:
            intermediate_features["output"] = []
        b, t, _, _, _ = x.shape
        t_emb = timestep_embedding(
            time_steps, self.model_channels, repeat_only=False
        ).type(x.dtype)
        emb = self.time_embed(t_emb)

        # repeat t times for context [(b t) 77 768] & time embedding
        # check if we use per-frame image conditioning
        _, l_context, _ = context.shape
        if l_context == 77 + t * 16:  # !!! HARD CODE here
            context_text, context_img = context[:, :77, :], context[:, 77:, :]
            context_text = context_text.repeat_interleave(repeats=t, dim=0)
            context_img = rearrange(context_img, "b (t l) c -> (b t) l c", t=t)
            context = torch.cat([context_text, context_img], dim=1)
        else:
            context = context.repeat_interleave(repeats=t, dim=0)
        emb = emb.repeat_interleave(repeats=t, dim=0)

        # always in shape (b t) c h w, except for temporal layer
        x = rearrange(x, "b t c h w -> (b t) c h w")

        # combine emb
        if self.fs_condition:
            if fs is None:
                fs = torch.tensor(
                    [self.default_fs] * b, dtype=torch.long, device=x.device
                )
            fs_emb = timestep_embedding(
                fs, self.model_channels, repeat_only=False
            ).type(x.dtype)

            fs_embed = self.fps_embedding(fs_emb)
            fs_embed = fs_embed.repeat_interleave(repeats=t, dim=0)
            emb = emb + fs_embed

        if self.camera_pose_condition:
            # camera_poses: (b, t, 12)
            camera_poses = rearrange(camera_poses, "b t x y -> (b t) (x y)")  # x=3, y=4
            camera_poses_embed = self.camera_pose_embedding(camera_poses)
            emb = emb + camera_poses_embed

        if self.use_task_embedding:
            assert (
                task_idx is not None
            ), "`task_idx` should not be None when `use_task_embedding` is enabled."
            task_embed = self.task_embedding(
                self.task_parameters[task_idx]
                .reshape(1, self.model_channels)
                .repeat(b, 1)
            )
            task_embed = task_embed.repeat_interleave(repeats=t, dim=0)
            emb = emb + task_embed

        h = x.type(self.dtype)
        adapter_idx = 0
        hs = []
        for _id, module in enumerate(self.input_blocks):

            h = module(h, emb, context=context, batch_size=b)
            if _id == 0 and self.addition_attention:
                h = self.init_attn(h, emb, context=context, batch_size=b)
            # plug-in adapter features
            if ((_id + 1) % 3 == 0) and features_adapter is not None:
                h = h + features_adapter[adapter_idx]
                adapter_idx += 1
            hs.append(h)
            if return_input_block_features:
                intermediate_features["input"].append(h)
        if features_adapter is not None:
            assert len(features_adapter) == adapter_idx, "Wrong features_adapter"

        h = self.middle_block(h, emb, context=context, batch_size=b)

        if return_middle_feature:
            intermediate_features["middle"] = h

        if self.use_feature_alignment:
            feature_alignment_output = self.feature_alignment_adapter(
                hs[2], hs[5], hs[8], emb=emb
            )

        # >>> Output Blocks Forward
        if self.use_ray_decoder:
            h_original = h
            h_ray = h
            for original_module, ray_module in zip(
                self.output_blocks, self.ray_decoder_blocks
            ):
                cur_hs = hs.pop()
                h_original = torch.cat([h_original, cur_hs], dim=1)
                h_original = original_module(
                    h_original,
                    emb,
                    context=context,
                    batch_size=b,
                    time_steps=time_steps,
                )
                if self.use_ray_decoder_residual:
                    h_ray = torch.cat([h_ray, cur_hs], dim=1)
                h_ray = ray_module(h_ray, emb, context=context, batch_size=b)
                if return_output_block_features:
                    print(
                        "return_output_block_features: h_original.shape=",
                        h_original.shape,
                    )
                    intermediate_features["output"].append(h_original.detach())
            h_original = h_original.type(x.dtype)
            h_ray = h_ray.type(x.dtype)
            y_original = self.out(h_original)
            y_ray = self.ray_output_head(h_ray)
            y = torch.cat([y_original, y_ray], dim=1)
        else:
            if self.use_lora_for_rays_in_output_blocks:
                middle_h = h
                h_original = middle_h
                h_lora = middle_h
                for output_idx, module in enumerate(self.output_blocks):
                    cur_hs = hs.pop()
                    h_original = torch.cat([h_original, cur_hs], dim=1)
                    h_original = module(
                        h_original, emb, context=context, batch_size=b, with_lora=False
                    )

                    h_lora = torch.cat([h_lora, cur_hs], dim=1)
                    h_lora = module(
                        h_lora, emb, context=context, batch_size=b, with_lora=True
                    )
                h_original = h_original.type(x.dtype)
                h_lora = h_lora.type(x.dtype)
                y_original = self.out(h_original)
                y_lora = self.ray_output_head(h_lora)
                y = torch.cat([y_original, y_lora], dim=1)
            else:
                for module in self.output_blocks:
                    h = torch.cat([h, hs.pop()], dim=1)
                    h = module(h, emb, context=context, batch_size=b)
                h = h.type(x.dtype)

                if self.use_task_embedding:
                    # Seperated Input (Branch Control in CPU)
                    # Serial Execution (GPU Vectorization Pending)
                    if task_idx == TASK_IDX_IMAGE:
                        y = self.out(h)
                    elif task_idx == TASK_IDX_RAY:
                        y = self.ray_output_head(h)
                    else:
                        raise NotImplementedError(f"Unsupported `task_idx`: {task_idx}")
                else:
                    # Output ray and images at the same forward
                    y = self.out(h)

                    if self.use_addition_ray_output_head:
                        y_ray = self.ray_output_head(h)
                        y = torch.cat([y, y_ray], dim=1)
        # reshape back to (b c t h w)
        y = rearrange(y, "(b t) c h w -> b t c h w", b=b)
        if (
            return_input_block_features
            or return_output_block_features
            or return_middle_feature
        ):
            return y, intermediate_features
        # Assume intermediate features are only request during non-training scenarios (e.g., feature visualization)
        if self.use_feature_alignment:
            return y, feature_alignment_output
        return y


class FeatureAlignmentAdapter(torch.nn.Module):
    def __init__(self, time_embed_dim, use_checkpoint, dropout=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.channel_adapter_conv_16 = torch.nn.Conv2d(
            in_channels=1280, out_channels=320, kernel_size=1
        )
        self.channel_adapter_conv_32 = torch.nn.Conv2d(
            in_channels=640, out_channels=320, kernel_size=1
        )
        self.upsampler_x2 = torch.nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsampler_x4 = torch.nn.UpsamplingBilinear2d(scale_factor=4)
        self.res_block = ResBlock(
            320 * 3,
            time_embed_dim,
            dropout,
            out_channels=32 * 3,
            dims=2,
            use_checkpoint=use_checkpoint,
            use_scale_shift_norm=False,
        )
        self.final_conv = conv_nd(
            dims=2, in_channels=32 * 3, out_channels=6, kernel_size=1
        )

    def forward(self, feature_64, feature_32, feature_16, emb):
        feature_16_adapted = self.channel_adapter_conv_16(feature_16)
        feature_32_adapted = self.channel_adapter_conv_32(feature_32)
        feature_16_upsampled = self.upsampler_x4(feature_16_adapted)
        feature_32_upsampled = self.upsampler_x2(feature_32_adapted)
        feature_all = torch.concat(
            [feature_16_upsampled, feature_32_upsampled, feature_64], dim=1
        )

        # bt, 3, h, w
        return self.final_conv(self.res_block(feature_all, emb=emb))
