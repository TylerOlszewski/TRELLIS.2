from typing import *
from contextlib import contextmanager
import torch
from .samplers import FlowEulerSampler


@contextmanager
def inject_multi_image_conditioning(
    sampler: FlowEulerSampler,
    num_images: int,
    num_steps: Optional[int] = None,
    mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
):
    """
    Temporarily patch a sampler so that a batch of image conditions (one per view
    of the same object) can drive a single sample.

    Args:
        sampler (FlowEulerSampler): The sampler to patch.
        num_images (int): The number of conditioning images.
        num_steps (int): The number of sampling steps (used to warn when views outnumber steps).
        mode (str): How to aggregate the image conditions.
            - 'stochastic': use a different image at each denoising step (round-robin).
            - 'multidiffusion': average the flow predictions of all images at every step.
    """
    if mode == 'stochastic':
        if num_steps is not None and num_images > num_steps:
            print(f"\033[93mWarning: number of conditioning images ({num_images}) exceeds the number of "
                  f"sampling steps ({num_steps}). Some images will not be used.\033[0m")

        call_count = [0]
        old_inference_model = sampler._inference_model

        def _new_inference_model(self, model, x_t, t, cond, **kwargs):
            cond_idx = call_count[0] % num_images
            call_count[0] += 1
            cond_i = cond[cond_idx:cond_idx + 1]
            return old_inference_model(model, x_t, t, cond=cond_i, **kwargs)

    elif mode == 'multidiffusion':
        def _new_inference_model(self, model, x_t, t, cond, neg_cond=None, guidance_strength=1.0,
                                 guidance_interval=(0.0, 1.0), guidance_rescale=0.0, **kwargs):
            preds = [
                FlowEulerSampler._inference_model(self, model, x_t, t, cond[i:i + 1], **kwargs)
                for i in range(num_images)
            ]
            pred_pos = preds[0]
            for p in preds[1:]:
                pred_pos = pred_pos + p
            pred_pos = pred_pos / num_images

            if not (guidance_interval[0] <= t <= guidance_interval[1]) or guidance_strength == 1:
                return pred_pos
            if guidance_strength == 0:
                return FlowEulerSampler._inference_model(self, model, x_t, t, neg_cond, **kwargs)

            pred_neg = FlowEulerSampler._inference_model(self, model, x_t, t, neg_cond, **kwargs)
            pred = pred_pos * guidance_strength + pred_neg * (1 - guidance_strength)

            # CFG rescale (mirrors ClassifierFreeGuidanceSamplerMixin)
            if guidance_rescale > 0:
                x_0_pos = self._pred_to_xstart(x_t, t, pred_pos)
                x_0_cfg = self._pred_to_xstart(x_t, t, pred)
                std_pos = x_0_pos.std(dim=list(range(1, x_0_pos.ndim)), keepdim=True)
                std_cfg = x_0_cfg.std(dim=list(range(1, x_0_cfg.ndim)), keepdim=True)
                x_0_rescaled = x_0_cfg * (std_pos / std_cfg)
                x_0 = x_0_rescaled * guidance_rescale + x_0_cfg * (1 - guidance_rescale)
                pred = self._xstart_to_pred(x_t, t, x_0)

            return pred

    else:
        raise ValueError(f"Unsupported multi-image mode: {mode}")

    sampler._inference_model = _new_inference_model.__get__(sampler, type(sampler))
    try:
        yield
    finally:
        del sampler._inference_model
