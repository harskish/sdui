import torch
from torch.nn import functional as F
from pathlib import Path
import sys
from . import k_diffusion as K

class CFGMaskedDenoiser(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale, mask, x0): #, xi):
        # Inplace modification
        # => visible for caller
        # if mask is not None:
        #     assert x0 is not None
        #     assert not x.requires_grad, 'Inplace ops break grads'
        #     img_orig = x0 + sigma * torch.randn_like(x0)
        #     x.copy_(x * (1 - mask) + img_orig * mask)
            
        #     # TEST
        #     x[-1, -1, -1, -1].copy_(torch.tensor(1337, dtype=x.dtype, device=x.device))
        
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

        return denoised

class KDiffusionSampler:
    def __init__(self, m, sampler):
        self.model = m
        self.model_wrap = K.external.CompVisDenoiser(m)
        self.schedule = sampler
    
    def get_sampler_name(self):
        return self.schedule
    
    # txt2img in masked regions, then img2img after chosen threshold
    def sample_general(
        self,
        steps_tot,          # number of total steps
        steps_img2img,      # number of steps for performing img2img at the end
        c,                  # c, cond from prompt
        guidance_scale,     # classifier free guidance scale
        uc,                 # uc, cond from empty prompt
        x_T,                # final noised latent, pure Gaussian noise
        x0=None,            # initial noise-free latent if out-/inpainting, optional
        mask=None,          # mask if out-/inpainting, optional
        img_callback=None,  # callback for showing progress
    ):
        assert steps_tot > 0 and steps_img2img >= 0, 'Invalid step counts'
        assert steps_img2img <= steps_tot, 'steps_img2img cannot be larger than total number of steps'
        
        # TODO: supply list of random numbers for whole diffusion process
        
        steps_txt2img = steps_tot - steps_img2img
        model_wrap_cfg = CFGMaskedDenoiser(self.model_wrap)
        func = getattr(K.sampling, f'sample_{self.schedule}')
        sigmas = self.model_wrap.get_sigmas(steps_tot).to(c.dtype) # linear resampling of original sigmas + final zero
        x = x_T * sigmas[0]

        def callback(vals: dict):
            i = vals['i']
            
            # Inplace modification
            # => visible to caller
            if mask is not None and i < steps_txt2img:
                x = vals['x']
                assert not x.requires_grad, 'Inplace ops break grads'
                assert x0 is not None
                img_orig = x0 + vals['sigma'] * torch.randn_like(x0)
                x.copy_(x * mask + img_orig * (1 - mask))
                #x[-1, -1, -1, -1].mul_(0) # TEST
            
            if img_callback:
                img_callback(vals['denoised'], vals['i'])
        

        # Skip ahead if pure unmasked img2img
        if steps_txt2img > 0 and mask is None and x0 is not None:
            sigmas = sigmas[(steps_tot - steps_img2img - 1):]
            x = x0 + x_T * sigmas[0]
            print(f'Skipping {steps_txt2img} steps txt2img, running {steps_img2img} steps img2img')
        else:
            print(f'Running {steps_txt2img} steps txt2img + {steps_img2img} steps img2img')
        
        # Run txt2img first if in-/outpainting, then run img2img
        x = func(model_wrap_cfg, x, sigmas, disable=False, callback=callback,
            extra_args={'cond': c, 'uncond': uc, 'cond_scale': guidance_scale, 'mask': mask, 'x0': x0})

        return x, None
    
    def sample(self, S, conditioning, batch_size, shape, verbose, unconditional_guidance_scale, unconditional_conditioning, eta, x_T, img_callback, mask=None, x0=None):
        sigmas = self.model_wrap.get_sigmas(S).to(conditioning.dtype)
        x = x_T * sigmas[0]
        model_wrap_cfg = CFGMaskedDenoiser(self.model_wrap)

        def cbk(vals: dict):
            img_callback(vals['denoised'], vals['i'])

        func = getattr(K.sampling, f'sample_{self.schedule}')
        samples_ddim = func(model_wrap_cfg, x, sigmas, disable=False, callback=cbk,
            extra_args={'cond': conditioning, 'uncond': unconditional_conditioning, 'cond_scale': unconditional_guidance_scale, 'mask': mask, 'x0': x0})

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
            extra_args={'cond': cond, 'uncond': unconditional_conditioning, 'cond_scale': unconditional_guidance_scale, 'mask': mask, 'x0': x0}) #'xi': xi

        return samples_ddim