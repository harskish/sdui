import torch
import numpy as np
from . import utils, external

from PIL import Image
from torch import autocast
from einops import rearrange, repeat
from tqdm import trange

"""
Find noise that reproduces input image
Currently implemented only for k_euler
Source: https://gist.github.com/trygvebw/c71334dd127d537a15e9d59790f7f5e1
"""

def pil_img_to_torch(pil_img):
    image = np.array(pil_img).astype(np.float32) / 255.0
    image = rearrange(torch.from_numpy(image), 'h w c -> c h w')
    return (2.0 * image - 1.0).unsqueeze(0)

def pil_img_to_latent(model, img, batch_size=1, device='cuda', dtype=torch.float16):
    init_image = pil_img_to_torch(img).to(device)
    init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
    return model.get_first_stage_encoding(model.encode_first_stage(init_image.to(dtype))).to(dtype)

def find_noise_for_image(model, img, prompt, steps=35, cond_scale=0.0, verbose=False, normalize=True, device='cuda', dtype=torch.float16):
    x = pil_img_to_latent(model, img, batch_size=1, device=device, dtype=dtype)

    with torch.no_grad():
        uncond = model.get_learned_conditioning(['']).to(dtype)
        cond = model.get_learned_conditioning([prompt]).to(dtype)

    s_in = x.new_ones([x.shape[0]])
    dnw = external.CompVisDenoiser(model)
    sigmas = dnw.get_sigmas(steps).flip(0).to(dtype)

    if verbose:
        print(sigmas)

    with torch.no_grad():
        for i in trange(1, len(sigmas), desc='Reverse sampling'):
            if cond_scale > 0:
                x_in = torch.cat([x] * 2)
                sigma_in = torch.cat([sigmas[i - 1] * s_in] * 2)
                cond_in = torch.cat([uncond, cond])

                c_out, c_in = [utils.append_dims(k, x_in.ndim) for k in dnw.get_scalings(sigma_in)]
                
                if i == 1:
                    t = dnw.sigma_to_t(torch.cat([sigmas[i] * s_in] * 2))
                else:
                    t = dnw.sigma_to_t(sigma_in)
                    
                eps = model.apply_model(x_in * c_in, t, cond=cond_in)
                denoised_uncond, denoised_cond = (x_in + eps * c_out).chunk(2)
                
                denoised = denoised_uncond + (denoised_cond - denoised_uncond) * cond_scale
            else:
                sigma_in = sigmas[i - 1] * s_in
                c_out, c_in = [utils.append_dims(k, x.ndim) for k in dnw.get_scalings(sigma_in)]
                
                if i == 1:
                    t = dnw.sigma_to_t(sigmas[i] * s_in)
                else:
                    t = dnw.sigma_to_t(sigma_in)
                    
                eps = model.apply_model(x * c_in, t, cond=uncond)
                denoised = x + eps * c_out
            
            if i == 1:
                d = (x - denoised) / (2 * sigmas[i])
            else:
                d = (x - denoised) / sigmas[i - 1]

            dt = sigmas[i] - sigmas[i - 1]
            x = x + d * dt
        
        if normalize:
            return (x / x.std()) * sigmas[-1]
        else:
            return x