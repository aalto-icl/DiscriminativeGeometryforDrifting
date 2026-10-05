import math

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from distributed import gather_with_grad, rank


def log_kernel(query, support, temp):
    return -torch.cdist(query.float(), support.float(), p=2) / (temp * math.sqrt(query.shape[-1]))


def loo_log_density(log_k, offset):
    n_query, n_support = log_k.shape[1], log_k.shape[2]
    idx = torch.arange(n_query, device=log_k.device)
    log_k = log_k.clone()
    log_k[:, idx, offset + idx] = -torch.inf
    return torch.logsumexp(log_k, dim=-1) - math.log(n_support - 1)


def cross_log_density(log_k):
    return torch.logsumexp(log_k, dim=-1) - math.log(log_k.shape[-1])


def group_loss(real_q, fake_q, real_s, fake_s, temp, offset):
    score_real = loo_log_density(log_kernel(real_q, real_s, temp), offset) - cross_log_density(log_kernel(real_q, fake_s, temp))
    score_fake = cross_log_density(log_kernel(fake_q, real_s, temp)) - loo_log_density(log_kernel(fake_q, fake_s, temp), offset)
    return F.softplus(-score_real).mean() + F.softplus(score_fake).mean()


def encoder_kde_loss(real_groups, fake_groups, temps):
    offset = rank() * real_groups[0].shape[0]
    groups = []
    for real, fake in zip(real_groups, fake_groups):
        groups.append([x.transpose(0, 1).contiguous() for x in (real, fake, gather_with_grad(real), gather_with_grad(fake))])
    losses = []
    for temp in temps:
        per_group = [checkpoint(group_loss, *g, temp, offset, use_reentrant=False) for g in groups]
        losses.append(torch.stack(per_group).mean())
    return torch.stack(losses).mean()
