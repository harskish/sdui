import os
import imgui
import torch
import numpy as np
from textwrap import dedent
from sys import exit
from dataclasses import dataclass
from copy import deepcopy
from functools import lru_cache
from viewer.toolbar_viewer import ToolbarViewer
from viewer.utils import reshape_grid, combo_box_vals
from typing import Dict, Tuple, Union
from os import makedirs
from dataclasses import asdict
from pathlib import Path
from glfw import KEY_LEFT_SHIFT
import gdown
import glfw
from functools import partial
from PIL.PngImagePlugin import PngInfo, PngImageFile
import json
from PIL import Image

from omegaconf import OmegaConf
from torch import autocast
from contextlib import nullcontext
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.ddpm import LatentDiffusion
from viewer.single_image_viewer import draw as draw_debug

# Suppress CLIP warning
import transformers
transformers.logging.set_verbosity_error()

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
    id = '1F8R6C_63mM49vjYRoaAgMTtHRrwX3YhZ' # sd-v1-4.ckpt
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
    gdown.download(id=id, output=str(trg), quiet=False)
    assert trg.is_file(), 'DL failed!'

def load_model_from_config(config, ckpt, verbose=False):
    print(f'Loading model from {ckpt}')
    pl_sd = torch.load(ckpt, map_location='cpu')
    sd = pl_sd['state_dict']
    model = instantiate_from_config(config.model)
    setattr(model, 'ckpt', ckpt)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print('missing keys:')
        print(m)
    if len(u) > 0 and verbose:
        print('unexpected keys:')
        print(u)

    model.to(device)
    model.eval()

    if device == 'cuda':
        print('Using half precision globally')
        model.half() # TODO: effect on quality?

    # Switch to EMA weights
    if model.use_ema:
        model.model_ema.store(model.model.parameters())
        model.model_ema.copy_to(model.model)
    
    return model

@lru_cache()
def get_model(pkl):
    config = OmegaConf.load('configs/stable-diffusion/v1-inference.yaml')
    model = load_model_from_config(config, pkl)
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

    def init_sampler(self, model):
        stype = self.state.sampler_type
        if stype == 'plms':
            return PLMSSampler(model)
        elif stype == 'ddim':
            return DDIMSampler(model)
        else:
            raise RuntimeError('Unknown sampler type')

    def init_model(self, pkl) -> LatentDiffusion:
        model = get_model(pkl)

        # Reset caches
        prev = self.rend.model
        if not prev or model.ckpt != prev.ckpt:
            self.rend.img_cache = {}
            self.rend.cond_cache = {}

        return model

    @property
    def to_dict(self):
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

        # Shapes
        C = 4
        f = 8
        shape = [C, s.H // f, s.W // f] # [C, H/f, W/f]; f = ds, c = latent channels

        # Perform computation
        # Detect changes
        # Only works for fields annotated with type (e.g. sliders: list)
        if self.rend.last_ui_state != s:
            self.rend.last_ui_state = s
            self.rend.model = self.init_model(s.pkl)
            self.rend.sampler = self.init_sampler(self.rend.model)
            self.rend.i = 0
            self.rend.intermed = torch.zeros(s.B, 3, s.H, s.W, device=device)

        # Check if work is done
        if self.rend.i >= s.T:
            return None

        model = self.rend.model

        def cache_key(s, i: int):
            meta = asdict(s)
            meta['seed'] += i
            return json.dumps(meta)

        # Read from or write to cache
        finished = [False]*s.B
        for i, img in enumerate(self.rend.intermed):
            key = cache_key(s, i)
            if key in self.rend.img_cache:
                self.rend.intermed[i] = torch.tensor(self.rend.img_cache[key], device=device)
                finished[i] = True
        
        # No need to compute?
        if all(finished):
            self.rend.i = s.T
            grid = reshape_grid(self.rend.intermed) # => HWC
            grid = grid if grid.device.type == 'cuda' else grid.cpu().numpy()
            return grid

        # Initial noise
        keys = [(s.seed + i, s.H, s.W) for i in range(s.B)]
        start_code = seeds_to_samples(keys, (len(keys), *shape)).to(device)
        
        class UserAbort(Exception):
            pass

        try:
            precision_scope = autocast if device != 'mps' else nullcontext
            with precision_scope(device):
                # Get conditioning
                prompt = s.prompt.replace('\n', ' ')
                cond_key = (prompt, s.guidance_scale)
                if cond_key not in self.rend.cond_cache:
                    uc = None if s.guidance_scale == 1.0 else model.get_learned_conditioning(s.B * [""])
                    c = model.get_learned_conditioning(s.B * [prompt])
                    self.rend.cond_cache[cond_key] = (uc, c)
                uc, c = self.rend.cond_cache[cond_key]

                def cbk_img(img_curr, i):
                    self.rend.i = i + 1
                    if s != self.state or glfw.window_should_close(self.v._window):
                        raise UserAbort
                    
                    # Always show after last iter
                    p = self.state_soft.preview_interval
                    if self.rend.i >= s.T or (p > 0 and i % p == 0):
                        x_samples_ddim = model.decode_first_stage(img_curr)
                        self.rend.intermed = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0) # [1, 3, 512, 512]
                        grid = reshape_grid(self.rend.intermed) # => HWC
                        grid = grid if grid.device.type == 'cuda' else grid.cpu().numpy()
                        self.v.upload_image(self.output_key, grid)
                        self.img_shape = self.rend.intermed.shape[1:]

                # Run image diffusion
                self.rend.sampler.sample(S=s.T, conditioning=c, batch_size=s.B,
                    shape=shape, verbose=False, unconditional_guidance_scale=s.guidance_scale,
                    unconditional_conditioning=uc, eta=0.0, x_T=start_code, img_callback=cbk_img)
        except UserAbort:
            # UI state changed, restart rendering
            return None

        # Write to cache
        for i, img in enumerate(self.rend.intermed):
            key = cache_key(s, i)
            if not (torch.any(torch.isnan(img)) or torch.all(img == 0)):
                self.rend.img_cache[key] = img.cpu().numpy()

        return None
    
    def draw_toolbar(self):
        jmp_large = 100 if self.v.keydown(KEY_LEFT_SHIFT) else 10

        s = self.state
        s.B = imgui.input_int('B', s.B)[1]
        s.seed = max(0, imgui.input_int('Seed', s.seed, s.B, 1)[1])
        s.T = imgui.input_int('T_img', s.T, 1, jmp_large)[1]
        s.H = combo_box_vals('H', list(range(64, 2048, 64)), s.H, to_str=str)[1]
        s.W = combo_box_vals('W', list(range(64, 2048, 64)), s.W, to_str=str)[1]
        scale_fmt = '%.2f' if np.modf(s.guidance_scale)[0] > 0 else '%.0f' # dynamically change from ints to floats
        s.sampler_type = combo_box_vals('Sampler', ['ddim', 'plms'], s.sampler_type)[1]
        s.guidance_scale = imgui.slider_float('Guidance', s.guidance_scale, 0, 20, format=scale_fmt)[1]
        self.state_soft.preview_interval = imgui.slider_int('Preview interval', self.state_soft.preview_interval, 0, 10)[1]
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
    seed: int = 0
    B: int = 1
    H: int = 512
    W: int = 512
    guidance_scale: float = 8.0 # classifier guidance
    sampler_type: str = 'ddim' # plms, ddim
    prompt: str = dedent('''
        東京, 吹雪, 夕方,
        National Geographic
        ''').strip()

# Non-volatile: changes don't force recomputation
@dataclass
class UIStateSoft:
    preview_interval: int = 5

@dataclass
class RendererState:
    last_ui_state: UIState = None # Detect changes in UI, restart rendering
    model: LatentDiffusion = None
    sampler: Union[PLMSSampler, DDIMSampler] = None
    intermed: torch.Tensor = None
    img_cache: Dict[str, torch.Tensor] = None
    cond_cache: Dict[Tuple[str, float], Tuple[torch.Tensor, torch.Tensor]] = None
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
