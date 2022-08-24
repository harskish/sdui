import os
import imgui
import torch
import argparse
import numpy as np
from multiprocessing import Lock
from dataclasses import dataclass
from copy import deepcopy
from functools import lru_cache
from viewer.toolbar_viewer import ToolbarViewer
from viewer.utils import reshape_grid, combo_box_vals
from typing import Dict, Tuple, Union
from os import makedirs
from pathlib import Path
from glfw import KEY_LEFT_SHIFT
from tqdm import trange
from pytorch_lightning import seed_everything

from omegaconf import OmegaConf
from torch import autocast
from contextlib import contextmanager, nullcontext
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.ddpm import LatentDiffusion

from viewer.single_image_viewer import draw as draw_debug

#args = None

# Choose backend
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
mps = getattr(torch.backends, 'mps', None)
if mps and mps.is_available() and mps.is_built():
    device = 'mps'
    _orig = torch.Tensor.__repr__
    torch.Tensor.__repr__ = lambda t: _orig(t.cpu()) # nightlies still slightly buggy...

def sample_seeds(N, base=None):
    if base is None:
        base = np.random.randint(np.iinfo(np.int32).max - N)
    return [(base + s) for s in range(N)]

def sample_normal(shape=(1, 512), seed=None):
    seeds = sample_seeds(shape[0], base=seed)
    return seeds_to_samples(seeds, shape)

def seeds_to_samples(seeds, shape=(1, 512)):
    latents = np.zeros(shape, dtype=np.float32)
    for i, seed in enumerate(seeds):
        rng = np.random.RandomState(seed)
        latents[i] = rng.standard_normal(shape[1:])
    
    return torch.tensor(latents)

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    setattr(model, 'ckpt', ckpt)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.to(device)
    model.eval()
    return model

class ModelViz(ToolbarViewer):    
    def __init__(self, name, batch_mode=False, hidden=False):
        self.batch_mode = batch_mode
        super().__init__(name, batch_mode=batch_mode, hidden=hidden)
    
    # Check for missing type annotations (break by-value comparisons)
    def check_dataclass(self, obj):
        for name in dir(obj):
            if name.startswith('__'):
                continue
            if name not in obj.__dataclass_fields__:
                raise RuntimeError(f'[ERR] Unannotated field: {name}')

    def setup_state(self):
        self.state = UIState()
        self.rend = RendererState()
        self.prompt_curr = self.state.prompt
        
        self.check_dataclass(self.state)
        self.check_dataclass(self.rend)
        self.G_lock = Lock()
    
    @lru_cache()
    def _get_model(self, pkl):
        config = OmegaConf.load('configs/stable-diffusion/v1-inference.yaml')
        model = load_model_from_config(config, pkl)
        return model

    def init_sampler(self, model):
        stype = self.state.sampler_type
        if stype == 'plms':
            return PLMSSampler(model)
        elif stype == 'ddim':
            return DDIMSampler(model)
        else:
            raise RuntimeError('Unknown sampler type')

    def init_model(self, pkl) -> LatentDiffusion:
        model = self._get_model(pkl)

        # Reset caches
        prev = self.rend.model
        if not prev or model.ckpt != prev.ckpt:
            self.rend.lat_cache = {}
            self.rend.img_cache = {}

        return model

    # Progress bar below images
    def draw_output_extra(self):
        self.rend.i = imgui.slider_int('', self.rend.i + 1, 1, self.rend.last_ui_state.T)[1] - 1

    def compute(self):
        # Copy for this frame
        s = deepcopy(self.state)

        # Perform computation
        # Detect changes
        # Only works for fields annotated with type (e.g. sliders: list)
        if self.rend.last_ui_state != s:
            self.rend.last_ui_state = s
            self.rend.model = self.init_model(s.pkl)
            self.rend.sampler = self.init_sampler(self.rend.model)
            self.rend.i = 0
            res = 512 #model.image_size?
            self.rend.intermed = sample_normal((s.B, 3, res, res), s.seed).to(device) # spaial noise
            # Conditioning based on prompt
            self.rend.c = self.rend.uc = None

        # Check if work is done
        if self.rend.i >= s.T - 1:
            return None

        model = self.rend.model

        ###############################
        # TEST: STABLEDIFF

        # Read from or write to cache
        finished = [False]*s.B
        for i, img in enumerate(self.rend.intermed):
            key = (s.prompt, s.seed + i, s.T)
            if key in self.rend.img_cache:
                self.rend.intermed[i] = torch.tensor(self.rend.img_cache[key], device=device)
                finished[i] = True
        
        # No need to compute?
        #if all(finished):
        #    return None

        # Shapes
        C = 4
        f = 8
        H = W = 512
        shape = [C, H // f, W // f] # [C, H/f, W/f]; f = ds, c = latent channels

        # Initial noise
        keys = [(s.seed + i) for i in range(s.B)]
        missing = [k for k in keys if k not in self.rend.lat_cache]
        if missing:
            noises = seeds_to_samples(missing, (len(missing), *shape)).to(device)
            for seed, lat in zip(missing, noises):
                self.rend.lat_cache[(seed)] = lat
        
        start_code = torch.stack([self.rend.lat_cache[k] for k in keys], dim=0)
        
        precision_scope = autocast if device != 'mps' else nullcontext
        with precision_scope(device):
            with model.ema_scope():
                # Get conditioning
                if self.rend.c is None:
                    if s.guidance_scale != 1.0:
                        self.rend.uc = model.get_learned_conditioning(s.B * [""])
                    self.rend.c = model.get_learned_conditioning(s.B * [s.prompt])

                def cbk_img(img_curr, i):
                    x_samples_ddim = model.decode_first_stage(img_curr)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0) # [1, 3, 512, 512]
                    grid = reshape_grid(x_samples_ddim) # => HWC
                    grid = grid if grid.device.type == 'cuda' else grid.cpu().numpy()
                    self.v.upload_image(self.output_key, grid)
                    self.rend.i += 1

                # Run image diffusion
                samples_ddim, _ = self.rend.sampler.sample(S=s.T, conditioning=self.rend.c, batch_size=s.B,
                                                    shape=shape,
                                                    verbose=False,
                                                    unconditional_guidance_scale=s.guidance_scale,
                                                    unconditional_conditioning=self.rend.uc,
                                                    eta=0.0, # ddim_eta, 0 = deterministic
                                                    x_T=start_code,
                                                    img_callback=cbk_img)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0) # [1, 3, 512, 512]

                self.rend.intermed = x_samples_ddim


        ################################



        # Run diffusion one step forward
        # T = torch.tensor([s.T] * s.B, device=model.dev_img).view(-1, 1)
        # t = T - self.rend.i - 1 # 0-based index, num_steps -> 0
        # if model.img_fused:
        #     self.rend.intermed = model.sample_img_incr_fused(T, t, self.rend.intermed, cond)
        # else:
        #     self.rend.intermed = model.sample_img_incr(t, self.rend.intermed, cond, *self.rend.img_samp_params)
        
        # Move on to next iteration
        #self.rend.i += 1
        self.rend.i = s.T # stop

        # Write to cache
        for i, img in enumerate(self.rend.intermed):
            key = (s.prompt, s.seed + i, s.T)
            if not torch.any(torch.isnan(img)): # MPS bug: sometimes contains NaNs that darken image
                self.rend.img_cache[key] = img.cpu().numpy()

        # Early exit
        if all(finished):
            self.rend.i = s.T - 1
        
        # Output updated grid
        grid = reshape_grid(self.rend.intermed) # => HWC
        return grid if grid.device.type == 'cuda' else grid.cpu().numpy()
    
    def draw_toolbar(self):
        jmp_large = 100 if self.v.keydown(KEY_LEFT_SHIFT) else 10

        s = self.state
        s.B = imgui.input_int('B', s.B)[1]
        s.seed = max(0, imgui.input_int('Seed', s.seed, s.B, 1)[1])
        s.T = imgui.input_int('T_img', s.T, 1, jmp_large)[1]
        self.prompt_curr = imgui.input_text_multiline('Prompt', self.prompt_curr, buffer_length=2048)[1]
        if imgui.button('Update'):
            s.prompt = self.prompt_curr.replace('\n', ' ')


# Volatile state: requires recomputation of results
@dataclass
class UIState:
    pkl: str = 'models/ldm/stable-diffusion-v1/model.ckpt'
    T: int = 35
    seed: int = 8
    B: int = 1
    guidance_scale: float = 7.5 # classifier guidance
    sampler_type: str = 'plms' # plms, ddim
    prompt: str = 'A cyberpunk rubber duck swimming in money at a night club with sun glasses and a cigar. Neon lights, bokeh, dof, f 1/2.2, octane render, macro'

@dataclass
class RendererState:
    last_ui_state: UIState = None # Detect changes in UI, restart rendering
    model: LatentDiffusion = None
    #img_samp_params: Dict[str, torch.Tensor] = None
    sampler: Union[PLMSSampler, DDIMSampler] = None
    intermed: torch.Tensor = None
    img_cache: Dict[Tuple[bool, int, int, int], torch.Tensor] = None
    lat_cache: Dict[Tuple[int, int], torch.Tensor] = None
    i: int = 0 # current computation progress

def init_torch():
    # Go fast
    torch.autograd.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True
    
    # Stay safe
    os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

if __name__ == '__main__':
    init_torch()
    viewer = ModelViz('sdiff_viewer', hidden=False)
    print('Done')
