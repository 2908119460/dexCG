"""PyTorch-native indexed reductions used by PartField."""

import torch


def _expanded_index(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    return index.expand_as(source) if index.shape != source.shape else index


def scatter_mean(
    source: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: torch.Tensor | None = None,
    dim_size: int | None = None,
) -> torch.Tensor:
    index = _expanded_index(source, index)
    if out is None:
        shape = list(source.shape)
        shape[dim] = dim_size if dim_size is not None else int(index.max()) + 1
        out = source.new_zeros(shape)
    out.scatter_add_(dim, index, source)
    counts = source.new_zeros(out.shape)
    counts.scatter_add_(dim, index, torch.ones_like(source))
    return out / counts.clamp_min(1)


def scatter_max(
    source: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: torch.Tensor | None = None,
    dim_size: int | None = None,
) -> tuple[torch.Tensor, None]:
    index = _expanded_index(source, index)
    if out is None:
        shape = list(source.shape)
        shape[dim] = dim_size if dim_size is not None else int(index.max()) + 1
        out = source.new_full(shape, torch.finfo(source.dtype).min)
    out.scatter_reduce_(dim, index, source, reduce="amax", include_self=True)
    return out, None

