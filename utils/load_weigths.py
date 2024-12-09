from utils.utils import instantiate_from_config
import torch
import copy
from omegaconf import OmegaConf
import logging

main_logger = logging.getLogger("main_logger")


def expand_conv_kernel(pretrained_dict):
    """expand 2d conv parameters from 4D -> 5D"""
    for k, v in pretrained_dict.items():
        if v.dim() == 4 and not k.startswith("first_stage_model"):
            v = v.unsqueeze(2)
            pretrained_dict[k] = v
    return pretrained_dict


def print_state_dict(state_dict):
    print("====== Dumping State Dict ======")
    for k, v in state_dict.items():
        print(k, v.shape)


def load_from_pretrainedSD_checkpoint(
    model,
    pretained_ckpt,
    expand_to_3d=True,
    adapt_keyname=False,
    echo_empty_params=False,
):
    sd_state_dict = torch.load(pretained_ckpt, map_location="cpu")
    if "state_dict" in list(sd_state_dict.keys()):
        sd_state_dict = sd_state_dict["state_dict"]
    model_state_dict = model.state_dict()
    # delete ema_weights just for <precise param counting>
    for k in list(sd_state_dict.keys()):
        if k.startswith("model_ema"):
            del sd_state_dict[k]
    main_logger.info(
        f"Num of model params of Source:{len(sd_state_dict.keys())} VS. Target:{len(model_state_dict.keys())}"
    )
    # print_state_dict(model_state_dict)
    # print_state_dict(sd_state_dict)

    if adapt_keyname:
        # adapting to standard 2d network: modify the key name because of the add of temporal-attention
        mapping_dict = {
            "middle_block.2": "middle_block.3",
            "output_blocks.5.2": "output_blocks.5.3",
            "output_blocks.8.2": "output_blocks.8.3",
        }
        cnt = 0
        for k in list(sd_state_dict.keys()):
            for src_word, dst_word in mapping_dict.items():
                if src_word in k:
                    new_key = k.replace(src_word, dst_word)
                    sd_state_dict[new_key] = sd_state_dict[k]
                    del sd_state_dict[k]
                    cnt += 1
        main_logger.info(f"[renamed {cnt} Source keys to match Target model]")

    pretrained_dict = {
        k: v for k, v in sd_state_dict.items() if k in model_state_dict
    }  # drop extra keys
    empty_paras = [
        k for k, v in model_state_dict.items() if k not in pretrained_dict
    ]  # log no pretrained keys
    assert len(empty_paras) + len(pretrained_dict.keys()) == len(
        model_state_dict.keys()
    )

    if expand_to_3d:
        # adapting to 2d inflated network
        pretrained_dict = expand_conv_kernel(pretrained_dict)

    # overwrite entries in the existing state dict
    model_state_dict.update(pretrained_dict)

    # load the new state dict
    try:
        model.load_state_dict(model_state_dict)
    except:
        skipped = []
        model_dict_ori = model.state_dict()
        for n, p in model_state_dict.items():
            if p.shape != model_dict_ori[n].shape:
                # skip by using original empty paras
                model_state_dict[n] = model_dict_ori[n]
                main_logger.info(
                    f"Skip para: {n}, size={pretrained_dict[n].shape} in pretrained, {model_state_dict[n].shape} in current model"
                )
                skipped.append(n)
        main_logger.info(
            f"[INFO] Skip {len(skipped)} parameters becasuse of size mismatch!"
        )
        model.load_state_dict(model_state_dict)
        empty_paras += skipped

    # only count Unet  part of depth estimation model
    unet_empty_paras = [
        name for name in empty_paras if name.startswith("model.diffusion_model")
    ]
    main_logger.info(
        f"Pretrained parameters: {len(pretrained_dict.keys())} | Empty parameters: {len(empty_paras)} [Unet:{len(unet_empty_paras)}]"
    )
    if echo_empty_params:
        print("Printing empty parameters:")
        for k in empty_paras:
            print(k)
    return model, empty_paras


# Below: written by Yingqing --------------------------------------------------------


def load_model_from_config(config, ckpt, verbose=False):
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        main_logger.info("missing keys:")
        main_logger.info(m)
    if len(u) > 0 and verbose:
        main_logger.info("unexpected keys:")
        main_logger.info(u)

    model.eval()
    return model


def init_and_load_ldm_model(config_path, ckpt_path, device=None):
    assert config_path.endswith(".yaml"), f"config_path = {config_path}"
    assert ckpt_path.endswith(".ckpt"), f"ckpt_path = {ckpt_path}"
    config = OmegaConf.load(config_path)
    model = load_model_from_config(config, ckpt_path)
    if device is not None:
        model = model.to(device)
    return model


def load_img_model_to_video_model(
    model,
    device=None,
    expand_to_3d=True,
    adapt_keyname=False,
    config_path="configs/latent-diffusion/txt2img-1p4B-eval.yaml",
    ckpt_path="models/ldm/text2img-large/model.ckpt",
):
    pretrained_ldm = init_and_load_ldm_model(config_path, ckpt_path, device)
    model, empty_paras = load_partial_weights(
        model,
        pretrained_ldm.state_dict(),
        expand_to_3d=expand_to_3d,
        adapt_keyname=adapt_keyname,
    )
    return model, empty_paras


def load_partial_weights(
    model, pretrained_dict, expand_to_3d=True, adapt_keyname=False
):
    model2 = copy.deepcopy(model)
    model_dict = model.state_dict()
    model_dict_ori = copy.deepcopy(model_dict)

    main_logger.info(f"[Load pretrained LDM weights]")
    main_logger.info(
        f"Num of parameters of source model:{len(pretrained_dict.keys())} VS. target model:{len(model_dict.keys())}"
    )

    if adapt_keyname:
        # adapting to menghan's standard 2d network: modify the key name because of the add of temporal-attention
        mapping_dict = {
            "middle_block.2": "middle_block.3",
            "output_blocks.5.2": "output_blocks.5.3",
            "output_blocks.8.2": "output_blocks.8.3",
        }
        cnt = 0
        newpretrained_dict = copy.deepcopy(pretrained_dict)
        for k, v in newpretrained_dict.items():
            for src_word, dst_word in mapping_dict.items():
                if src_word in k:
                    new_key = k.replace(src_word, dst_word)
                    pretrained_dict[new_key] = v
                    pretrained_dict.pop(k)
                    cnt += 1
        main_logger.info(f"--renamed {cnt} source keys to match target model.")
    pretrained_dict = {
        k: v for k, v in pretrained_dict.items() if k in model_dict
    }  # drop extra keys
    empty_paras = [
        k for k, v in model_dict.items() if k not in pretrained_dict
    ]  # log no pretrained keys
    main_logger.info(
        f"Pretrained parameters: {len(pretrained_dict.keys())} | Empty parameters: {len(empty_paras)}"
    )
    # disable info
    # main_logger.info(f'Empty parameters: {empty_paras} ')
    assert len(empty_paras) + len(pretrained_dict.keys()) == len(model_dict.keys())

    if expand_to_3d:
        # adapting to yingqing's 2d inflation network
        pretrained_dict = expand_conv_kernel(pretrained_dict)

    # overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)

    # load the new state dict
    try:
        model2.load_state_dict(model_dict)
    except:
        # if parameter size mismatch, skip them
        skipped = []
        for n, p in model_dict.items():
            if p.shape != model_dict_ori[n].shape:
                # skip by using original empty paras
                model_dict[n] = model_dict_ori[n]
                main_logger.info(
                    f"Skip para: {n}, size={pretrained_dict[n].shape} in pretrained, {model_dict[n].shape} in current model"
                )
                skipped.append(n)
        main_logger.info(
            f"[INFO] Skip {len(skipped)} parameters becasuse of size mismatch!"
        )
        model2.load_state_dict(model_dict)
        empty_paras += skipped
        main_logger.info(f"Empty parameters: {len(empty_paras)} ")

    main_logger.info(f"Finished.")
    return model2, empty_paras


def load_autoencoder(model, config_path=None, ckpt_path=None, device=None):
    if config_path is None:
        config_path = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
    if ckpt_path is None:
        ckpt_path = "models/ldm/text2img-large/model.ckpt"

    pretrained_ldm = init_and_load_ldm_model(config_path, ckpt_path, device)
    autoencoder_dict = {}
    for n, p in pretrained_ldm.state_dict().items():
        if n.startswith("first_stage_model"):
            autoencoder_dict[n] = p
    model_dict = model.state_dict()
    model_dict.update(autoencoder_dict)
    main_logger.info(f"Load [{len(autoencoder_dict)}] autoencoder parameters!")

    model.load_state_dict(model_dict)

    return model
