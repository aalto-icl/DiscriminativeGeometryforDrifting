import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).to(x.dtype)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, dropout):
        super().__init__()
        self.norm1 = GroupNorm32(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch * 2))
        self.norm2 = GroupNorm32(32, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.time_proj(temb)[:, :, None, None].chunk(2, dim=1)
        h = F.silu(self.norm2(h) * (1 + scale) + shift)
        h = self.conv2(self.dropout(h))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, ch, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.norm = GroupNorm32(32, ch)
        self.qkv = nn.Conv1d(ch, ch * 3, 1)
        self.out = nn.Conv1d(ch, ch, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x).reshape(b, c, h * w))
        q, k, v = qkv.reshape(b, 3, self.num_heads, c // self.num_heads, h * w).unbind(1)
        out = F.scaled_dot_product_attention(q.transpose(2, 3), k.transpose(2, 3), v.transpose(2, 3))
        out = out.transpose(2, 3).reshape(b, c, h * w)
        return x + self.out(out).reshape(b, c, h, w)


class ResAttnBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, dropout, use_attn, num_heads):
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, time_dim, dropout)
        self.attn = SelfAttention(out_ch, num_heads) if use_attn else None

    def forward(self, x, temb):
        x = self.res(x, temb)
        return self.attn(x) if self.attn is not None else x


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base_ch=128, ch_mult=(1, 2, 2, 2), num_res_blocks=2,
                 attn_resolutions=(16,), dropout=0.1, num_heads=4, image_size=32):
        super().__init__()
        self.in_ch = in_ch
        self.image_size = image_size
        self.base_ch = base_ch
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        time_dim = base_ch * 4
        self.time_embed = nn.Sequential(nn.Linear(base_ch, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.conv_in = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        skip_chs = [base_ch]
        ch = base_ch
        res = image_size
        for level, mult in enumerate(ch_mult):
            out = base_ch * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResAttnBlock(ch, out, time_dim, dropout, res in attn_resolutions, num_heads))
                ch = out
                skip_chs.append(ch)
            if level < len(ch_mult) - 1:
                self.down_samples.append(Downsample(ch))
                skip_chs.append(ch)
                res //= 2

        self.mid_block1 = ResAttnBlock(ch, ch, time_dim, dropout, True, num_heads)
        self.mid_block2 = ResAttnBlock(ch, ch, time_dim, dropout, False, num_heads)

        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        for level in reversed(range(len(ch_mult))):
            out = base_ch * ch_mult[level]
            for _ in range(num_res_blocks + 1):
                self.up_blocks.append(
                    ResAttnBlock(ch + skip_chs.pop(), out, time_dim, dropout, res in attn_resolutions, num_heads)
                )
                ch = out
            if level > 0:
                self.up_samples.append(Upsample(ch))
                res *= 2

        self.norm_out = GroupNorm32(32, ch)
        self.conv_out = nn.Conv2d(ch, out_ch, 3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x):
        t = torch.zeros(x.shape[0], device=x.device, dtype=torch.long)
        temb = self.time_embed(timestep_embedding(t, self.base_ch))
        h = self.conv_in(x)
        skips = [h]
        block = 0
        for level in range(len(self.ch_mult)):
            for _ in range(self.num_res_blocks):
                h = self.down_blocks[block](h, temb)
                skips.append(h)
                block += 1
            if level < len(self.ch_mult) - 1:
                h = self.down_samples[level](h)
                skips.append(h)
        h = self.mid_block2(self.mid_block1(h, temb), temb)
        block = 0
        for i, level in enumerate(reversed(range(len(self.ch_mult)))):
            for _ in range(self.num_res_blocks + 1):
                h = self.up_blocks[block](torch.cat([h, skips.pop()], dim=1), temb)
                block += 1
            if level > 0:
                h = self.up_samples[i](h)
        return self.conv_out(F.silu(self.norm_out(h)))
