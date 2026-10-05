import math

import torch

from distributed import all_max, all_sum, gather, rank, world_size


def column_softmax(logit):
    col_max = all_max(logit.max(dim=1).values)
    exp = torch.exp(logit - col_max.unsqueeze(1))
    return exp / all_sum(exp.sum(dim=1)).unsqueeze(1)


def drift_field(gen, pos, neg, temp, self_offset):
    n_query, n_pos, dim = gen.shape[1], pos.shape[1], gen.shape[2]
    support = torch.cat([pos, neg], dim=1).float()
    dist = torch.cdist(gen.float(), support, p=2)
    idx = torch.arange(n_query, device=gen.device)
    dist[:, idx, n_pos + self_offset + idx] += 1e6
    logit = -dist / (temp * math.sqrt(dim))
    a = torch.sqrt(torch.softmax(logit, dim=2) * column_softmax(logit) + 1e-30)
    a_pos, a_neg = a[:, :, :n_pos], a[:, :, n_pos:]
    w_pos = a_pos * a_neg.sum(dim=2, keepdim=True)
    w_neg = a_neg * a_pos.sum(dim=2, keepdim=True)
    return torch.bmm(w_pos, support[:, :n_pos]) - torch.bmm(w_neg, support[:, n_pos:])


def drift_loss(gen_groups, pos_groups, temps, v_max):
    ws = world_size()
    total = 0.0
    for gen, pos in zip(gen_groups, pos_groups):
        local_batch, _, dim = gen.shape
        with torch.no_grad():
            query = gen.detach().transpose(0, 1).contiguous()
            pos_all = gather(pos.detach()).transpose(0, 1).contiguous()
            neg_all = gather(gen.detach()).transpose(0, 1).contiguous()
            mean_dist = torch.cdist(query.float(), torch.cat([pos_all, neg_all], dim=1).float(), p=2).mean()
            scale = (all_sum(mean_dist) / ws / math.sqrt(dim)).clamp(min=1e-8)
            query, pos_all, neg_all = query / scale, pos_all / scale, neg_all / scale
            velocity = torch.zeros_like(query)
            for temp in temps:
                v = drift_field(query, pos_all, neg_all, temp, rank() * local_batch)
                rms = torch.sqrt(all_sum(v.float().pow(2).sum() / v.numel()) / ws)
                velocity = velocity + v / (rms + 1e-8)
            if v_max > 0:
                particle_rms = velocity.float().pow(2).mean(dim=-1).sqrt()
                velocity = velocity * (v_max / (particle_rms + 1e-8)).clamp(max=1.0).unsqueeze(-1)
            target = (query + velocity).transpose(0, 1)
        total = total + (gen / scale - target).pow(2).mean(dim=(0, 2)).sum()
    return total
