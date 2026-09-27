import torch


def torch_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(*shape, device=device) + lower


@torch.jit.script
def tensor_clamp(t, min_t, max_t):
    return torch.max(torch.min(t, max_t), min_t)


__all__ = ["tensor_clamp", "torch_rand_float"]
