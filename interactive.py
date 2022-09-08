import os
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

import torch
import imgui
import numpy as np
from textwrap import dedent
from sys import exit
from dataclasses import dataclass
from copy import deepcopy
from functools import lru_cache
from viewer.toolbar_viewer import ToolbarViewer
from viewer.utils import reshape_grid, combo_box_vals
from typing import Dict, Tuple, Union, List
from os import makedirs
from dataclasses import asdict
from contextlib import nullcontext
from base64 import b64encode, b64decode
from pathlib import Path
from glfw import KEY_LEFT_SHIFT
import gdown
import glfw
import hashlib
from functools import partial
from PIL.PngImagePlugin import PngInfo, PngImageFile
from pytorch_lightning import seed_everything
from multiprocessing import Lock
import json
from PIL import Image

from mps_patches import apply_mps_patches
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.k_samplers import KDiffusionSampler
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.modules.encoders.modules import get_default_device_type
from ldm.modules import attention
#from viewer.single_image_viewer import draw as draw_debug

SAMPLERS_IMG2IMG = ['ddim', 'k_dpm_2_a', 'k_dpm_2', 'k_euler_a', 'k_euler', 'k_heun', 'k_lms']
SAMPLERS_ALL = SAMPLERS_IMG2IMG + ['plms']

# Suppress CLIP warning
import transformers
transformers.logging.set_verbosity_error()

# Float16 patch
from ldm.modules.diffusionmodules.util import GroupNorm32
from torch.nn.functional import group_norm
def forward(self, x):
    return group_norm(x.float(), self.num_groups,
        self.weight.float(), self.bias.float(), self.eps).type(x.dtype)
GroupNorm32.forward = forward

# Choose backend
device = get_default_device_type()
if device == 'mps':
    apply_mps_patches()

def file_drop_callback(window, paths, viewer):
    # imgui.get_mouse_pose() sometimes returns -1...
    posx = glfw.get_cursor_pos(window)[0] # relative to window top left
    hovering_img = posx > viewer.toolbar_width
    
    for p in paths:
        if hovering_img:
            # Hovering over image => load as UI state
            viewer.load_state_from_img(p)
        else:
            # Hovering over settings bar => load as conditioning
            if viewer.rend.model is None:
                print('Model not loaded, please try again later')
            else:
                img = Image.open(p)
                viewer.rend.cond_img_orig = img
                viewer.get_cond_from_img(img, viewer.state.image_cond_scale_mode)

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

# FP16: float list ~6x larger than base64-encoded one when json-serialized
def arr_to_b64(arr: np.ndarray):
    arr = arr.astype(np.dtype(arr.dtype).newbyteorder('<')) # convert to little-endian
    b64_str = b64encode(arr.reshape(-1).tobytes()).decode('ascii') # little-endian bytes to base64 str
    return b64_str

def b64_to_arr(s: str, dtype: np.dtype):
    dtype = np.dtype(dtype).newbyteorder('<') # incoming data is little-endian
    buffer = b64decode(s.encode('ascii'))
    elems = len(buffer) // dtype.itemsize
    recovered = np.ndarray(shape=elems, dtype=dtype, buffer=buffer)
    return recovered.astype(dtype.newbyteorder('=')) # to native byte order

# Hash string into uint32
@np.errstate(over='ignore')
def djb2_hash(s: str):
    hash = np.uint32(5381)
    for c in s:
        hash = np.uint32(33) * hash + np.uint32(ord(c))
    return int(hash)

# Imgui slider that can switch between int and float formatting at runtime
def slider_dynamic(title, v, min, max):
    scale_fmt = '%.2f' if np.modf(v)[0] > 0 else '%.0f' # dynamically change from ints to floats
    return imgui.slider_float(title, v, min, max, format=scale_fmt)

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

def load_model_from_config(config, ckpt, use_half=False, verbose=False):
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

    if use_half:
        model.half()

    model.to(device)
    model.eval()

    # Switch to EMA weights
    if model.use_ema:
        model.model_ema.store(model.model.parameters())
        model.model_ema.copy_to(model.model)
    
    return model

def get_act_shape(state):
    return [state.C, state.H // state.f, state.W // state.f]

@lru_cache()
def get_model(pkl, use_half):
    config = OmegaConf.load('configs/stable-diffusion/v1-inference.yaml')
    
    #print(config['model']['params']['first_stage_config'])
    #print(config['model']['params']['unet_config'])
    #print(config['model']['params']['cond_stage_config'])
    config['model']['params']['unet_config']['params']['use_fp16'] = use_half
    
    model = load_model_from_config(config, pkl, use_half)
    
    # Does not seem to support float16
    model.cond_stage_model.float()

    return model

class ModelViz(ToolbarViewer):    
    def __init__(self, name, input=None):
        self.input = input
        super().__init__(name, batch_mode=False, hidden=False)

    @property
    def dtype(self):
        return torch.float16 if self.state.fp16 else torch.float32
    
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
        self.state_lock = Lock()
        self.prompt_curr = self.state.prompt        
        self.check_dataclass(self.state)
        self.check_dataclass(self.state_soft)
        self.check_dataclass(self.rend)

        if self.input:
            self.load_state_from_img(self.input)

        if device == 'cpu':
            self.state.fp16 = False

        self.post_init()

    def init_sampler(self, model):
        stype = self.state.sampler_type
        if stype == 'plms':
            return PLMSSampler(model)
        elif stype == 'ddim':
            return DDIMSampler(model)
        elif stype == 'k_dpm_2_a':
            return KDiffusionSampler(model, 'dpm_2_ancestral')
        elif stype == 'k_dpm_2':
            return KDiffusionSampler(model, 'dpm_2')
        elif stype == 'k_euler_a':
            return KDiffusionSampler(model, 'euler_ancestral')
        elif stype == 'k_euler':
            return KDiffusionSampler(model, 'euler')
        elif stype == 'k_heun':
            return KDiffusionSampler(model, 'heun')
        elif stype == 'k_lms':
            return KDiffusionSampler(model, 'lms')
        else:
            raise RuntimeError('Unknown sampler type')

    def init_model(self, pkl) -> LatentDiffusion:
        model = get_model(pkl, self.state.fp16)

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

    def load_state_from_img(self, path):
        suff = Path(path).suffix
        if suff != '.png':
            print('Cannot load UI state from non-png files')
            return
        
        meta = get_meta_from_img(path)
        self.from_dict(meta)

    def reshape_image_cond(self, need_lock=True):
        if self.state.image_cond is None:
            return
        
        # Original not available, cannot rescale
        if self.rend.cond_img_orig is None:
            # https://stackoverflow.com/a/31179482
            n_bytes = (len(self.state.image_cond) * 3) // 4 - self.state.image_cond.count('=', -2)
            n_elems_fp16 = n_bytes // 2
            if n_elems_fp16 != np.prod(get_act_shape(self.state)):
                print('Image conditioning shape not compatible, removing...')
                self.state.image_cond = None
        else:
            # Adapt image to current shape
            self.get_cond_from_img(self.rend.cond_img_orig, self.state.image_cond_scale_mode, need_lock=need_lock)

    def from_dict(self, state_dict_in):
        self.state_lock.acquire()

        state_dict = state_dict_in['state']
        state_dict_soft = state_dict_in['state_soft']
        
        # Ignore certain values
        ignores = ['fp16']
        state_dict = { k: v for k,v in state_dict.items() if k not in ignores }
        state_dict_soft = { k: v for k,v in state_dict_soft.items() if k not in ignores }
    
        for k, v in state_dict.items():
            setattr(self.state, k, v)
        
        for k, v in state_dict_soft.items():
            setattr(self.state_soft, k, v)

        # Convert old-style image-cond
        if isinstance(self.state.image_cond, list):
            self.state.image_cond = arr_to_b64(np.array(self.state.image_cond).astype(np.float16))
        
        # If updating image conditioning: invalidate preview image
        if 'image_cond' in state_dict:
            self.rend.cond_img_handle = None

        self.post_init()
        
        self.state_lock.release()

    # After init or state load
    def post_init(self):
        self.state_soft.attn_group_size = attention.ATTN_GROUP_SIZE = \
            min(attention.ATTN_GROUP_SIZE, self.state_soft.attn_group_size)
        self.reshape_image_cond()
        self.prompt_curr = self.state.prompt

    def export_img(self):
        grid = reshape_grid(self.rend.intermed).contiguous() # HWC
        im = Image.fromarray(np.uint8(255*grid.clip(0,1).cpu().numpy()))
        metadata = json.dumps(self.to_dict, sort_keys=True)
        
        from datetime import datetime
        fname = datetime.now().strftime(r'%Y%m%d_%H%M%S.png')
        outdir = Path('./outputs')
        os.makedirs(outdir, exist_ok=True)

        chunk = PngInfo()
        chunk.add_text('description', metadata)
        opt = np.prod(grid.shape[:2]) < 2_000_000
        im.save(outdir / fname, format='png', optimize=opt, compress_level=9, pnginfo=chunk) # max compression
        
        print('Saved as', fname)

    # Load image, get conditioning info
    # Mutates state, results in recompute
    # Scale modes:
    # - stretch: ignore aspect ratio, resample
    # - center: unscaled masked outpaint on mag, center-crop on min
    # - fit: resize keeping AR, then masked outpaint
    def get_cond_from_img(self, image_in: Image, scale_mode: str, need_lock=True):
        image = image_in.copy().convert('RGB')
        W, H = self.state.W, self.state.H
        w, h = image.size

        bg_color = imgui.get_style().colors[imgui.COLOR_WINDOW_BACKGROUND][0:3]
        canvas = Image.new(image.mode, (W, H), tuple(map(lambda c: int(255*c), bg_color)))

        if (W, H) == (w, h):
            pass # no processing needed
        elif scale_mode == 'center':
            pass # no processing needed
        elif scale_mode == 'fit':
            scale = min(H / h, W / w) # smaller scale along either dim
            image = image.resize((int(round(w*scale)), int(round(h*scale))), resample=Image.Resampling.LANCZOS)
        elif scale_mode == 'stretch':
            image = image.resize((W, H), resample=Image.Resampling.LANCZOS)
        else:
            raise ValueError(f'Unknown scaling mode {scale_mode}')

        # Paste centered
        w, h = image.size # potentially changed
        top_left = ((W - w) // 2, (H - h) // 2)
        canvas.paste(image, top_left)

        np_img = np.array(canvas).astype(np.float32) / 255.0
        np_img = np_img[None].transpose(0, 3, 1, 2)
        init_image = 2 * torch.tensor(np_img.copy()).to(self.dtype).to(device) - 1

        # Encoder produces (mean, logvar) that parametrize diagonal gaussian, which is sampled
        # Smaller output => ill-posed, need distribution
        encoded = self.rend.model.encode_first_stage(init_image)
        encoded.std *= 0 # use mean directly, no variance
        init_latent = self.rend.model.get_first_stage_encoding(encoded)  # computes (mean + std * randn(...))*scale_factor
        
        _, C, H_per_f, W_per_f = init_latent.shape
        assert C == self.state.C, 'C does not match'

        # Set mask if needed
        if w < W or h < H:
            lef_x, top_y = top_left
            rig_x, bot_y = (lef_x + w, top_y + h)
            
            # Conservative valid image ranges
            lef_x = int(np.ceil(lef_x / self.state.f))
            rig_x = int(np.floor(rig_x / self.state.f))
            top_y = int(np.ceil(top_y / self.state.f))
            bot_y = int(np.floor(bot_y / self.state.f))

            mask = torch.zeros((1, 1, H_per_f, W_per_f), device=device, dtype=self.dtype)
            mask[:, :, top_y:bot_y, lef_x:rig_x] = 1
            self.rend.cond_img_mask = 1 - mask  # forces inner part to stay unchanged
            
            # Use random init for outside parts
            # Not needed? This is the final latent which is noised anyway later
            #init_latent = mask * init_latent + (1 - mask) * torch.randn_like(init_latent) * encoded.std * self.rend.model.scale_factor
        else:
            self.rend.cond_img_mask = None
        
        # Make sure state consistent
        with (self.state_lock if need_lock else nullcontext()):
            # Not all samplers support img2img
            if self.state.sampler_type not in SAMPLERS_IMG2IMG:
                self.state.sampler_type = 'ddim'
            self.state.H = H_per_f * self.state.f
            self.state.W = W_per_f * self.state.f
            self.state.image_cond = arr_to_b64(init_latent.cpu().numpy().astype(np.float16))

            # For faster hashing of state
            self.state.image_cond_hash = hashlib.sha1(np_img.tobytes()).hexdigest()
            print('Cond image hash:', self.state.image_cond_hash)

        import random, string
        handle = self.rend.cond_img_handle or ''.join(random.choices(string.ascii_letters, k=20))
        self.v.upload_image(handle, np.array(canvas)) # full res image shown
        self.rend.cond_img_handle = handle

    # Use current output as conditioning input
    def load_cond_from_current(self):
        out_np = self.rend.intermed[0].cpu().numpy().transpose(1, 2, 0) # HWC
        img = Image.fromarray(np.uint8(255*out_np))
        self.rend.cond_img_orig = img
        self.get_cond_from_img(img, self.state.image_cond_scale_mode)

    def compute(self):
        # Copy for this frame
        with self.state_lock:
            s = deepcopy(self.state)

        # Shape of activations before SR network
        shape = get_act_shape(s)

        # Perform computation
        # Detect changes
        # Only works for fields annotated with type (e.g. sliders: list)
        if self.rend.last_ui_state != s:
            self.rend.last_ui_state = s
            self.rend.model = self.init_model(s.pkl)
            self.rend.sampler = self.init_sampler(self.rend.model)
            self.rend.i = 0
            self.rend.intermed = torch.zeros(s.B, 3, s.H, s.W, device=device, dtype=self.dtype)

        # Check if work is done
        if self.rend.i >= s.T:
            return None

        model = self.rend.model
        
        # Compute hash of state
        state_hash = hashlib.sha1(json.dumps(asdict(s)).encode('utf-8')).hexdigest()
        def cache_key(i: int):
            return state_hash + str(i)

        # Read from or write to cache
        finished = [False]*s.B
        for i, img in enumerate(self.rend.intermed):
            key = cache_key(i)
            if key in self.rend.img_cache:
                self.rend.intermed[i] = torch.tensor(self.rend.img_cache[key], device=device)
                finished[i] = True
        
        # No need to compute?  
        if all(finished):
            self.rend.i = s.T
            grid = reshape_grid(self.rend.intermed) # => HWC
            grid = grid if grid.device.type == 'cuda' else grid.cpu().numpy()
            return grid
        
        class UserAbort(Exception):
            pass

        try:
            # Get conditioning
            prompt = s.prompt.replace('\n', ' ')
            cond_key = (prompt, s.guidance_scale)
            if cond_key not in self.rend.cond_cache:
                uc = None if s.guidance_scale == 1.0 else model.get_learned_conditioning([""]).to(self.dtype)
                c = model.get_learned_conditioning([prompt]).to(self.dtype)
                self.rend.cond_cache[cond_key] = (uc, c)
            uc, c = map(lambda v: v.repeat((s.B, 1, 1)) if v is not None else v, self.rend.cond_cache[cond_key])

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
                    H, W, C = grid.shape
                    self.img_shape = (C, H, W)
            
            if s.image_cond is not None:
                # img2img
                x0_np = b64_to_arr(s.image_cond, np.float16).reshape([1, *shape])
                x0 = torch.tensor(x0_np, device=device).repeat((s.B, 1, 1, 1)).to(self.dtype)
                strength = 1 - (s.image_cond_strength - 1) / 10 # [1, 10] => [0, 9] => [1.0, 0.1]
                t_enc = max(2, int(strength * s.T)) # less than two breaks image

                # Image suffers if using same noise in encode and cond image generation
                # => make sure sequences differ
                # TODO: need to seed whole diffusion process, not just initial rand
                base = djb2_hash(s.image_cond_hash) # hash loaded from state dump => deterministic
                seeds = [(base + s.seed + i) % (1<<32-1) for i in range(s.B)]
                noises = seeds_to_samples(seeds, (len(seeds), *shape)).to(device).to(self.dtype)

                # TODO: instead provide cross-platform randn_like to samplers as callback
                seed_everything(s.seed)
                
                if self.rend.cond_img_mask is not None:
                    # Outpainting using text2img
                    seeds = [s.seed + i for i in range(s.B)]
                    start_code = seeds_to_samples(seeds, (len(seeds), *shape)).to(self.dtype).to(device)
                    self.rend.sampler.sample(
                        S=s.T,
                        conditioning=c,
                        batch_size=s.B,
                        shape=shape,
                        verbose=False,
                        unconditional_guidance_scale=s.guidance_scale,
                        unconditional_conditioning=uc,
                        eta=0.0, 
                        x_T=start_code,                  # latent at end of noising process (step T)
                        mask=1-self.rend.cond_img_mask,  # mask for out-/inpainting, inverted...!
                        x0=x0,                           # clean latent of cond image
                        img_callback=cbk_img)
                elif self.state.sampler_type == 'ddim':
                    # Standard img2img

                    # Add noise to clean latent based on starting point in diffusion process
                    # TODO: ddpm_model.q_sample() vs ddim_sampler.stochastic_encode()?
                    self.rend.sampler.make_schedule(s.T, ddim_eta=0.0, verbose=False)
                    z_enc = self.rend.sampler.stochastic_encode(x0, torch.tensor([t_enc]*s.B).to(device), noise=noises)
                    
                    # Run diffusion process from chosen starting point
                    self.rend.sampler.decode(z_enc.to(x0.dtype), c, t_enc, unconditional_conditioning=uc, 
                        unconditional_guidance_scale=s.guidance_scale, img_callback=cbk_img)
                else:
                    # Standard img2img
                    ret = self.rend.sampler.decode(x0, c, noises, t_enc, s.T, unconditional_conditioning=uc,
                        unconditional_guidance_scale=s.guidance_scale, mask=self.rend.cond_img_mask, img_callback=cbk_img)
                    cbk_img(ret, s.T - 1) # iteration exits early, show last image
            else:
                # txt2img
                # TODO: need to seed whole diffusion process, not just initial rand
                seeds = [s.seed + i for i in range(s.B)]
                start_code = seeds_to_samples(seeds, (len(seeds), *shape)).to(self.dtype).to(device)

                # Random initial noise
                seed_everything(s.seed)
                self.rend.sampler.sample(S=s.T, conditioning=c, batch_size=s.B,
                    shape=shape, verbose=False, unconditional_guidance_scale=s.guidance_scale,
                    unconditional_conditioning=uc, eta=0.0, x_T=start_code, img_callback=cbk_img)
        except UserAbort:
            # UI state changed, restart rendering
            return None

        # Write to cache
        for i, img in enumerate(self.rend.intermed):
            key = cache_key(i)
            if not (torch.any(torch.isnan(img)) or torch.all(img == 0)):
                self.rend.img_cache[key] = img.cpu().numpy()

        # Finished
        self.rend.i = s.T
        
        return None
    
    # Progress bar below images
    def draw_output_extra(self):
        self.rend.i = imgui.slider_int('', self.rend.i, 0, self.rend.last_ui_state.T)[1]

    def draw_toolbar(self):
        jmp_large = 100 if self.v.keydown(KEY_LEFT_SHIFT) else 10

        s = self.state
        s.B = imgui.input_int('B', s.B)[1]
        s.seed = max(0, imgui.input_int('Seed', s.seed, s.B, 1)[1])
        s.T = imgui.input_int('T', s.T, 1, jmp_large)[1]
        
        #self.state_lock.acquire()
        with self.state_lock:
            chH, s.H = combo_box_vals('H', list(range(64, 2048, 64)), s.H, to_str=str)
            chW, s.W = combo_box_vals('W', list(range(64, 2048, 64)), s.W, to_str=str)
            if chH or chW:
                self.reshape_image_cond(need_lock=False) # context manager lock released upon function call?
        #self.state_lock.release()
        
        # Speed-VRAM tradeoff, larger = faster
        ch, self.state_soft.attn_group_size = combo_box_vals('Attn. group size', [2, 4, 8, 16], self.state_soft.attn_group_size)
        if imgui.is_item_hovered():
            imgui.set_tooltip('Attention group size, smaller values reduce VRAM requirement at the cost of speed')
        if ch:
            torch.cuda.empty_cache()
            attention.ATTN_GROUP_SIZE = self.state_soft.attn_group_size

        s.sampler_type = combo_box_vals('Sampler', SAMPLERS_ALL if s.image_cond is None else SAMPLERS_IMG2IMG, s.sampler_type)[1]
        s.guidance_scale = slider_dynamic('Guidance', s.guidance_scale, 0, 20)[1]
        self.state_soft.preview_interval = imgui.slider_int('Preview interval', self.state_soft.preview_interval, 0, 10)[1]
        self.prompt_curr = imgui.input_text_multiline('Prompt', self.prompt_curr, buffer_length=2048)[1]
        if imgui.button('Update'):
            s.prompt = self.prompt_curr

        imgui.text('Conditioning image:')
        if self.state.image_cond is None:
            imgui.same_line()
            imgui.text('None')
        else:
            if self.rend.cond_img_handle is not None:
                self.v.draw_image(self.rend.cond_img_handle, width=self.ui_scale*180)
                ch, s.image_cond_scale_mode = combo_box_vals('Scale mode', ['fit', 'center', 'stretch'], s.image_cond_scale_mode)
                if ch:
                    self.reshape_image_cond()
            else:
                imgui.same_line()
                imgui.text('no preview available')
            s.image_cond_strength = slider_dynamic('Strength', s.image_cond_strength, 1, 10)[1]
            if imgui.button('Remove'):
                with self.state_lock:
                    self.rend.cond_img_handle = None
                    self.rend.cond_img_orig = None
                    s.image_cond = None
                    s.image_cond_hash = None
            imgui.same_line()
        if imgui.button('Use current'):
            self.load_cond_from_current()

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
    C: int = 4
    f: int = 8
    fp16: bool = True
    guidance_scale: float = 8.0 # classifier guidance
    sampler_type: str = 'k_euler'
    image_cond: str = None
    image_cond_strength: float = 7.0 # [0, 10]
    image_cond_hash: str = None
    image_cond_scale_mode: str = 'fit' # center, stretch
    prompt: str = dedent('''
        東京, 吹雪, 夕方,
        National Geographic
        ''').strip()

# Non-volatile: changes don't force recomputation
@dataclass
class UIStateSoft:
    preview_interval: int = 5
    attn_group_size: int = 16

@dataclass
class RendererState:
    last_ui_state: UIState = None # Detect changes in UI, restart rendering
    model: LatentDiffusion = None
    sampler: Union[PLMSSampler, DDIMSampler] = None
    intermed: torch.Tensor = None
    cond_img_handle: str = None # handle to GL texture of conditioning img
    cond_img_orig: Image = None # original conditioning image
    cond_img_mask: torch.Tensor = None
    img_cache: Dict[str, torch.Tensor] = None
    cond_cache: Dict[Tuple[str, float], Tuple[torch.Tensor, torch.Tensor]] = None
    i: int = 0 # current computation progress

def init_torch():
    # Go fast
    torch.autograd.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = False
    
    # Stay safe
    os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Stable Diffusion visualizer')
    parser.add_argument('input', type=str, nargs='?', default=None, help='Image to load state from')
    args = parser.parse_args()
    
    download_weights()
    init_torch()
    viewer = ModelViz('sdui', input=args.input)
    print('Done')
