import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def world_size():
    return dist.get_world_size() if is_distributed() else 1


def rank():
    return dist.get_rank() if is_distributed() else 0


def all_sum(x):
    if is_distributed():
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x


def all_max(x):
    if is_distributed():
        dist.all_reduce(x, op=dist.ReduceOp.MAX)
    return x


def gather(x):
    if not is_distributed():
        return x
    out = [torch.empty_like(x) for _ in range(world_size())]
    dist.all_gather(out, x.contiguous())
    return torch.cat(out, dim=0)


def gather_with_grad(x):
    if not is_distributed():
        return x
    return torch.cat(dist_nn.all_gather(x), dim=0)
