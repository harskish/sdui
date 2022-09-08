import torch
from pathlib import Path
import sys
from . import k_diffusion as K

class CFGDenoiser(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale):
        if uncond is None or cond_scale == 1:
            return self.inner_model(x, sigma, cond=cond)
        else:
            x_in = torch.cat([x] * 2)
            sigma_in = torch.cat([sigma] * 2)
            cond_in = torch.cat([uncond, cond])
            uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
            return uncond + (cond - uncond) * cond_scale

class CFGMaskedDenoiser(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale, mask, x0, xi):
        denoised = None
        if uncond is None or cond_scale == 1:
            denoised = self.inner_model(x, sigma, cond=cond)
        else:
            x_in = x
            x_in = torch.cat([x_in] * 2)
            sigma_in = torch.cat([sigma] * 2)
            cond_in = torch.cat([uncond, cond])
            uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
            denoised = uncond + (cond - uncond) * cond_scale

        if mask is not None:
            assert x0 is not None
            img_orig = x0
            mask_inv = 1. - mask
            denoised = (img_orig * mask_inv) + (mask * denoised)

        return denoised

class KDiffusionSampler:
    def __init__(self, m, sampler):
        self.model = m
        self.model_wrap = K.external.CompVisDenoiser(m)
        self.schedule = sampler
    
    def get_sampler_name(self):
        return self.schedule
    
    def sample(self, S, conditioning, batch_size, shape, verbose, unconditional_guidance_scale, unconditional_conditioning, eta, x_T, img_callback):
        sigmas = self.model_wrap.get_sigmas(S).to(conditioning.dtype)
        x = x_T * sigmas[0]
        model_wrap_cfg = CFGDenoiser(self.model_wrap)

        def cbk(vals: dict):
            img_callback(vals['denoised'], vals['i'])

        func = getattr(K.sampling, f'sample_{self.schedule}')
        samples_ddim = func(model_wrap_cfg, x, sigmas, disable=False, callback=cbk,
            extra_args={'cond': conditioning, 'uncond': unconditional_conditioning, 'cond_scale': unconditional_guidance_scale})

        return samples_ddim, None

    
    def decode(self, x_latent, cond, noises, S, S_tot, unconditional_guidance_scale=1.0, unconditional_conditioning=None, mask=None, img_callback=None):
        x0 = x_latent

        sigmas = self.model_wrap.get_sigmas(S_tot).to(x_latent.dtype)
        noise = noises * sigmas[S_tot - S - 1]

        xi = x0 + noise

        sigma_sched = sigmas[(S_tot - S - 1):]
        model_wrap_cfg = CFGMaskedDenoiser(self.model_wrap)
        
        def cbk(vals: dict):
            img_callback(vals['denoised'], vals['i'])
        
        func = getattr(K.sampling, f'sample_{self.schedule}')
        samples_ddim = func(model_wrap_cfg, xi, sigma_sched, disable=False, callback=cbk,
            extra_args={'cond': cond, 'uncond': unconditional_conditioning, 'cond_scale': unconditional_guidance_scale, 'mask': mask, 'x0': x0, 'xi': xi})

        return samples_ddim