import torch


def process_inference_batch(cfg_scale, batch, model, with_uncondition_extra=False):
    for k in batch.keys():
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(model.device, dtype=model.dtype)
    z, cond, x_rec = model.get_batch_input(
        batch,
        random_drop_training_conditions=False,
        return_reconstructed_target_images=True,
    )
    # batch_size = x_rec.shape[0]
    # Get unconditioned embedding for classifier-free guidance sampling
    if cfg_scale != 1.0:
        uc = model.get_unconditional_dict_for_sampling(batch, cond, x_rec)
    else:
        uc = None

    if with_uncondition_extra:
        uc_extra = model.get_unconditional_dict_for_sampling(
            batch, cond, x_rec, is_extra=True
        )
        return cond, uc, uc_extra, x_rec
    else:
        return cond, uc, x_rec
