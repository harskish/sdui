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
from typing import Dict, Tuple, Union, List
from os import makedirs
from dataclasses import asdict
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
from ldm.modules.attention import BasicTransformerBlock
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
        suff = Path(p).suffix.lower()
        if hovering_img:
            # Hovering over image => load as UI state
            viewer.load_state_from_img(p)
        else:
            # Hovering over settings bar => load as conditioning
            if viewer.rend.model is None:
                print('Model not loaded, please try again later')
            else:
                viewer.get_cond_from_img(Image.open(p))

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
    config['model']['params']['unet_config']['params']['use_checkpoint'] = False
    
    model = load_model_from_config(config, pkl, use_half)
    model.eval()

    def remove_checkpointing(model):
        if isinstance(model, BasicTransformerBlock):
            model.checkpoint = False
        for c in model.children():
            remove_checkpointing(c)
    
    remove_checkpointing(model)

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
        self.export_to_onnx(model)

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

    def reshape_image_cond(self):
        # Check that resolutions match
        if self.state.image_cond:
            # https://stackoverflow.com/a/31179482
            n_bytes = (len(self.state.image_cond) * 3) // 4 - self.state.image_cond.count('=', -2)
            n_elems_fp16 = n_bytes // 2
            if n_elems_fp16 != np.prod(get_act_shape(self.state)):
                print('Image conditioning shape not compatible, removing... (TODO: resize instead?)')
                self.state.image_cond = None

    def from_dict(self, state_dict_in):
        self.state_lock.acquire()

        state_dict = state_dict_in['state']
        state_dict_soft = state_dict_in['state_soft']
        
        # Ignore certain values
        ignores = ['fp16']
        state_dict = { k: v for k,v in state_dict.items() if k not in ignores }
    
        for k, v in state_dict.items():
            setattr(self.state, k, v)
        
        for k, v in state_dict_soft.items():
            setattr(self.state_soft, k, v)

        self.prompt_curr = self.state.prompt

        # Convert old-style image-cond
        if isinstance(self.state.image_cond, list):
            self.state.image_cond = arr_to_b64(np.array(self.state.image_cond).astype(np.float16))
        
        # If updating image conditioning: invalidate preview image
        if 'image_cond' in state_dict:
            self.rend.cond_img_handle = None

        self.reshape_image_cond()
        
        self.state_lock.release()

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
    def get_cond_from_img(self, image: Image):
        image = image.convert("RGB")
        w, h = image.size
        w, h = map(lambda x: x - x % 32, (w, h)) # resize to integer multiple of 32
        image = image.resize((w, h), resample=Image.LANCZOS)
        np_img = np.array(image).astype(np.float32) / 255.0
        np_img = np_img[None].transpose(0, 3, 1, 2)
        init_image = 2 * torch.tensor(np_img.copy()).to(self.dtype).to(device) - 1

        # TODO: MPS results broken due to UI thread?
        enc = self.rend.model.encode_first_stage(init_image)
        init_latent = self.rend.model.get_first_stage_encoding(enc)  # [1, 4, H//f, W//f]
        
        _, C, H_per_f, W_per_f = init_latent.shape
        assert C == self.state.C, 'C does not match'
        
        # Make sure state is not corrupt
        with self.state_lock:
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
        handle = ''.join(random.choices(string.ascii_letters, k=20))
        preview_img = image.resize((W_per_f, H_per_f), resample=Image.LANCZOS)
        self.v.upload_image(handle, np.array(preview_img))
        self.rend.cond_img_handle = handle

    # Use current output as conditioning input
    def load_cond_from_current(self):
        out_np = self.rend.intermed[0].cpu().numpy().transpose(1, 2, 0) # HWC
        img = Image.fromarray(np.uint8(255*out_np))
        self.get_cond_from_img(img)

    def export_to_onnx(self, model, force=False):
        dtype = { torch.float16: 'fp16', torch.float32: 'fp32' }[self.dtype]

        inner = model.model.diffusion_model
        inner.eval()
        
        for p in inner.parameters():
            p.requires_grad_(False)

        assert model.model.conditioning_key == 'crossattn'
        
        B = 1
        x_in = torch.randn(B, *get_act_shape(self.state), device=device, dtype=self.dtype) # [B, 4, 64, 64]
        t_in = torch.ones(B, 1, device=device, dtype=torch.int64) # [B, 1]
        c_in = torch.randn((B, 2, 77, 768), device=device, dtype=self.dtype) # [B, 2, 77, 768]
        ex_inputs = (x_in, t_in, c_in)
        input_names = ['x', 't', 'c']

        jitted = None
        for dyn_b, dyn_sz in [(False, False), (True, False), (True, True)]:
            desc = '_dynHW' if dyn_sz else f'_{self.state.H}x{self.state.W}'
            desc = f'{desc}_dynB' if dyn_b else f'{desc}_B{B}'
            fname = Path(f'ckpts/unet_{dtype}_{device}{desc}.onnx')
            if fname.is_file() and not force:
                print('Already exported, skipping')
                continue
            
            def fwd_fun(x: torch.Tensor, t: torch.Tensor, c: torch.Tensor):
                # Duplicate twice using same underlying memory
                x = x.repeat_interleave(2, dim=0)                 # [B, 4, 64, 64] => [B*2, 4, 64, 64]
                t = t.repeat_interleave(2, dim=0).view(-1)        # [B, 1] => [B*2]
                c = c.view(2 * c.shape[0], *c.shape[2:])          # [B, 2, 77, 768] => [B*2, 77, 768]
                ret = inner.forward(x, t, context=c)              # [B*2, 4, 64, 64]
                ret = ret.view(-1, 2, *get_act_shape(self.state)) # [B, 2, 4, 64, 64]
                return ret
                
            if jitted is None:
                print('Jitting')
                jitted = torch.jit.trace(fwd_fun, ex_inputs, check_trace=False)

            def get_dynamic_axes(dynamic_B=False, dynamic_res=False):
                if not (dynamic_B or dynamic_res):
                    return None
                
                axes = { 'x': {}, 't': {}, 'c': {}, 'et_uc_c': {} }
                if dynamic_B:
                    axes['x'][0] = 'batch_dim'
                    axes['t'][0] = 'batch_dim'
                    axes['c'][0] = 'batch_dim'
                    axes['et_uc_c'][0] = 'batch_dim'
                if dynamic_res:
                    axes['x'][2] = 'height'
                    axes['x'][3] = 'width'
                    axes['et_uc_c'][3] = 'height'
                    axes['et_uc_c'][4] = 'width'

                return axes

            print('Exporting')
            makedirs(fname.parent, exist_ok=True)
            torch.onnx.export(jitted, ex_inputs, fname,
                input_names=input_names, output_names=['et_uc_c'], opset_version=13, # einsum, dynamic rep._interl.
                do_constant_folding=False, dynamic_axes=get_dynamic_axes(dyn_b, dyn_sz))
            
            print('Exported as', fname)

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
            self.rend.intermed = torch.zeros(s.B, 3, s.H, s.W, device=device)

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
            uc, c = map(lambda v: v.repeat((s.B, 1, 1)), self.rend.cond_cache[cond_key])

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
                arr = b64_to_arr(s.image_cond, np.float16).reshape([1, *shape])
                t = torch.tensor(arr, device=device).repeat((s.B, 1, 1, 1)).to(self.dtype)
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
                if self.state.sampler_type == 'ddim':    
                    self.rend.sampler.make_schedule(s.T, ddim_eta=0.0, verbose=False)
                    z_enc = self.rend.sampler.stochastic_encode(t, torch.tensor([t_enc]*s.B).to(device), noise=noises)
                    self.rend.sampler.decode(z_enc.to(t.dtype), c, t_enc, unconditional_conditioning=uc, 
                        unconditional_guidance_scale=s.guidance_scale, img_callback=cbk_img)
                else:
                    self.rend.sampler.decode(t, c, noises, t_enc, s.T, unconditional_conditioning=uc,
                        unconditional_guidance_scale=s.guidance_scale, img_callback=cbk_img)
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
        with self.state_lock:
            s.H = combo_box_vals('H', list(range(64, 2048, 64)), s.H, to_str=str)[1]
            s.W = combo_box_vals('W', list(range(64, 2048, 64)), s.W, to_str=str)[1]
            self.reshape_image_cond()

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
            else:
                imgui.same_line()
                imgui.text('no preview available')
            s.image_cond_strength = slider_dynamic('Strength', s.image_cond_strength, 1, 10)[1]
            if imgui.button('Remove'):
                with self.state_lock:
                    self.rend.cond_img_handle = None
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
    cond_img_handle: str = None # handle to GL texture of conditioning img
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
    import argparse
    parser = argparse.ArgumentParser(description='Stable Diffusion visualizer')
    parser.add_argument('input', type=str, nargs='?', default=None, help='Image to load state from')
    args = parser.parse_args()
    
    download_weights()
    init_torch()
    viewer = ModelViz('sdui', input=args.input)
    print('Done')
