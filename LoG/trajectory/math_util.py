import torch


@torch.jit.script
def normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # channel last: normalization
    return x / (torch.norm(x, dim=-1, keepdim=True) + eps)

@torch.jit.script
def affine_padding(c2w: torch.Tensor):
    # Already padded
    if c2w.shape[-2] == 4:
        return c2w
    # Batch agnostic padding
    sh = c2w.shape
    pad0 = c2w.new_zeros(sh[:-2] + (1, 3))  # B, 1, 3
    pad1 = c2w.new_ones(sh[:-2] + (1, 1))  # B, 1, 1
    pad = torch.cat([pad0, pad1], dim=-1)  # B, 1, 4
    c2w = torch.cat([c2w, pad], dim=-2)  # B, 4, 4
    return c2w