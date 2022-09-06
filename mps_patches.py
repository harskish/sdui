def apply_mps_patches():
    import torch
    from torch.nn.functional import gelu
    from ldm.modules.attention import GEGLU
    from ldm.models.diffusion.k_diffusion import external, utils

    # Frac not available (yet, as of 1.9.2022)
    def patched(self, t):
        t = t.float()
        low_idx, high_idx = t.floor().long(), t.ceil().long()
        w = t - low_idx # t.frac()
        return (1 - w) * self.sigmas[low_idx] + w * self.sigmas[high_idx]
    external.DiscreteSchedule.t_to_sigma = patched

    # Sort not available
    orig = external.DiscreteSchedule.sigma_to_t
    def patched(self, sigma, quantize=None):
        self.sigmas = self.sigmas.cpu().float()
        return orig(self, sigma.cpu().float(), quantize).to(sigma.device).to(sigma.dtype)
    external.DiscreteSchedule.sigma_to_t = patched

    def forward(self, x):
        # MPS: no float16 gelu
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * gelu(gate.float()).to(x.dtype)
    GEGLU.forward = forward

    # arr[None] reshaping op broken
    def patched(x, target_dims):
        dims_to_append = target_dims - x.ndim
        if dims_to_append < 0:
            raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
        return x.reshape(x.shape + (1,) * dims_to_append)
    utils.append_dims = patched

    # Nightlies still slightly buggy...
    _orig = torch.Tensor.__repr__
    torch.Tensor.__repr__ = lambda t: _orig(t.cpu())

    print('MPS patches applied')