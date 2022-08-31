import torch
from pathlib import Path
import sys

kdiff_root = Path(__file__).parent / 'k-diffusion'
assert kdiff_root.is_dir(), 'Submodules missing; please run "git submodule update --init --recursive"'
sys.path += [str(kdiff_root)]
import k_diffusion as K # type: ignore

class CFGDenoiser(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        return uncond + (cond - uncond) * cond_scale

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

    
    def decode(self, x_latent, cond, noises, S, S_tot, unconditional_guidance_scale=1.0, unconditional_conditioning=None, img_callback=None):
        x0 = x_latent

        sigmas = self.model_wrap.get_sigmas(S_tot).to(x_latent.dtype)
        noise = noises * sigmas[S_tot - S - 1]

        xi = x0 + noise

        sigma_sched = sigmas[(S_tot - S - 1):]
        model_wrap_cfg = CFGDenoiser(self.model_wrap)
        
        def cbk(vals: dict):
            img_callback(vals['denoised'], vals['i'])
        
        func = getattr(K.sampling, f'sample_{self.schedule}')
        samples_ddim = func(model_wrap_cfg, xi, sigma_sched, disable=False, callback=cbk,
            extra_args={'cond': cond, 'uncond': unconditional_conditioning, 'cond_scale': unconditional_guidance_scale}) #'mask': None, 'x0': x0, 'xi': xi

        return samples_ddim