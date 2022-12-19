from pathlib import Path
import numpy as np
from base64 import b64encode, b64decode
import zlib
import imgui
import torch

def get_default_device_type():
    mps = getattr(torch.backends, "mps", None)
    if torch.cuda.is_available():
        return "cuda"
    elif mps and mps.is_available() and mps.is_built():
        return "mps"
    else:
        return "cpu"

# Strip python-style comments
def parse_urls(list: Path) -> list[str]:
    if not list.is_file():
        return []
    
    lines = []
    for l in list.read_text().splitlines():
        comment_idx = l.find('#')
        if comment_idx >= 0:
            l = l[0:comment_idx]
        l = l.strip()
        if l:
            lines.append(l)
    
    return lines

# FP16: float list ~6x larger than base64-encoded one when json-serialized
# Header: {compressed?},{ndim},{shape0}(,{shape1},...),{data}
def arr_to_b64(arr: np.ndarray, compress=True):
    header = f'{"c" if compress else "u"},{arr.ndim},' + ','.join(map(str, arr.shape))
    arr = arr.astype(np.dtype(arr.dtype).newbyteorder('<')) # convert to little-endian
    arr_bytes = zlib.compress(arr.tobytes()) if compress else arr.tobytes()
    b64_str = b64encode(arr_bytes).decode('ascii') # little-endian bytes to base64 str
    return header + ',' + b64_str

def b64_parse_header(s: str):
    mode, ndim, s = s.split(',', maxsplit=2)
    *shape, s = s.split(',', maxsplit=int(ndim))
    shape = tuple(int(v) for v in shape)
    return mode, shape, s

def b64_to_arr(s: str, dtype: np.dtype):
    mode, shape, s = b64_parse_header(s)
    buffer = b64decode(s.encode('ascii'))
    if mode == 'c':
        buffer = zlib.decompress(buffer)
    dtype = np.dtype(dtype).newbyteorder('<') # incoming data is little-endian
    recovered = np.ndarray(shape=shape, dtype=dtype, buffer=buffer)
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