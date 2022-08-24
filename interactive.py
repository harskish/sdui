import os
import imgui
import torch
import argparse
import numpy as np
from textwrap import dedent
from sys import exit
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
import gdown
import glfw
from functools import partial
from PIL.PngImagePlugin import PngInfo, PngImageFile
import json
from PIL import Image

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

def file_drop_callback(window, paths, viewer):
    for p in paths:
        suff = Path(p).suffix.lower()
        if suff == '.png':
            meta = get_meta_from_img(p)
            viewer.from_dict(meta)

def get_meta_from_img(path: str):
    state_dump = r'{}'
    
    if path.endswith('.png'):
        test = PngImageFile(path)
        state_dump = test.text['description']
    else:
        print(f'Unknown extension {Path(path).suffix}')

    return json.loads(state_dump)

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

def download_weights():
    #src = 'https://drive.google.com/file/d/1F8R6C_63mM49vjYRoaAgMTtHRrwX3YhZ' #/view?usp=sharing'
    trg = Path('models/ldm/stable-diffusion-v1/model.ckpt')
    if trg.is_file():
        return

    resp = None
    while resp not in ['yes', 'y', 'no', 'n']:
        resp = input(dedent(
        '''
        The model weights are licensed under the CreativeML OpenRAIL License.
        Please read the full license here: https://huggingface.co/spaces/CompVis/stable-diffusion-license
        Do you accept the terms? yes/no
        '''))
    
    if resp in ['no', 'n']:
        exit(-1)

    makedirs(trg.parent, exist_ok=True)
    gdown.download(id='1F8R6C_63mM49vjYRoaAgMTtHRrwX3YhZ', output=str(trg), quiet=False)
    assert trg.is_file(), 'DL failed!'

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

    def setup_callbacks(self, window):
        glfw.set_drop_callback(window,
            partial(file_drop_callback, viewer=self))
    
    # Check for missing type annotations (break by-value comparisons)
    def check_dataclass(self, obj):
        for name in dir(obj):
            if name.startswith('__'):
                continue
            if name not in obj.__dataclass_fields__:
                raise RuntimeError(f'[ERR] Unannotated field: {name}')

    def setup_state(self):
        self.state = UIState()
        self.state_soft = UIStateSoft()
        self.rend = RendererState()
        self.prompt_curr = self.state.prompt
        
        self.check_dataclass(self.state)
        self.check_dataclass(self.state_soft)
        self.check_dataclass(self.rend)
    
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

    @property
    def to_dict(self):
        from dataclasses import asdict
        return {
            'state': asdict(self.state),
            'state_soft': asdict(self.state_soft),
        }

    def from_dict(self, state_dict_in):
        state_dict = state_dict_in['state']
        state_dict_soft = state_dict_in['state_soft']
        
        # Ignore certain values
        ignores = []
        state_dict = { k: v for k,v in state_dict.items() if k not in ignores }
    
        for k, v in state_dict.items():
            setattr(self.state, k, v)
        
        for k, v in state_dict_soft.items():
            setattr(self.state_soft, k, v)

        self.prompt_curr = self.state.prompt

    def export_img(self):
        grid = reshape_grid(self.rend.intermed).contiguous() # HWC
        im = Image.fromarray(np.uint8(255*grid.clip(0,1).cpu().numpy()))
        metadata = json.dumps(self.to_dict, sort_keys=True)
        
        from datetime import datetime
        fname = datetime.now().strftime(r'%d%m%Y_%H%M%S.png')
        outdir = Path('./outputs')
        os.makedirs(outdir, exist_ok=True)

        chunk = PngInfo()
        chunk.add_text('description', metadata)
        opt = np.prod(grid.shape[:2]) < 2_000_000
        im.save(outdir / fname, format='png', optimize=opt, compress_level=9, pnginfo=chunk) # max compression
        
        print('Saved as', fname)

    # Progress bar below images
    def draw_output_extra(self):
        self.rend.i = imgui.slider_int('', self.rend.i, 0, self.rend.last_ui_state.T)[1]

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
                    self.rend.c = model.get_learned_conditioning(s.B * [s.prompt.replace('\n', ' ')])

                def cbk_img(img_curr, i):
                    if self.state_soft.show_preview:
                        x_samples_ddim = model.decode_first_stage(img_curr)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0) # [1, 3, 512, 512]
                        grid = reshape_grid(x_samples_ddim) # => HWC
                        grid = grid if grid.device.type == 'cuda' else grid.cpu().numpy()
                        self.v.upload_image(self.output_key, grid)
                    self.rend.i += 1

                # Run image diffusion
                samples_ddim, _ = self.rend.sampler.sample(S=s.T-1, conditioning=self.rend.c, batch_size=s.B,
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
        self.state_soft.show_preview = imgui.checkbox('Interactive preview', self.state_soft.show_preview)[1]
        self.prompt_curr = imgui.input_text_multiline('Prompt', self.prompt_curr, buffer_length=2048)[1]
        if imgui.button('Update'):
            s.prompt = self.prompt_curr

        if imgui.button('Export image'):
            self.export_img()


# Volatile state: requires recomputation of results
@dataclass
class UIState:
    pkl: str = 'models/ldm/stable-diffusion-v1/model.ckpt'
    T: int = 35
    seed: int = 2
    B: int = 1
    guidance_scale: float = 7.5 # classifier guidance
    sampler_type: str = 'plms' # plms, ddim
    prompt: str = dedent('''
        A guitar on fire,
        sunset, nebula
        ''').strip()

# Non-volatile: changes don't force recomputation
@dataclass
class UIStateSoft:
    show_preview: bool = True

@dataclass
class RendererState:
    last_ui_state: UIState = None # Detect changes in UI, restart rendering
    model: LatentDiffusion = None
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
    download_weights()
    init_torch()
    viewer = ModelViz('sdui', hidden=False)
    print('Done')
