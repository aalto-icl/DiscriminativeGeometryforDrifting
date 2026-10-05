import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.SiLU(),
        )

    def forward(self, x):
        return x + self.branch(x)


class ResidualEncoder(nn.Module):
    def __init__(self, channels=3, hidden_channels=128, dilations=(1, 2, 4, 1), pool_size=4):
        super().__init__()
        self.pool_size = pool_size
        self.residual = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 3, padding=1),
            nn.SiLU(),
            *[ResidualBlock(hidden_channels, d) for d in dilations],
            nn.Conv2d(hidden_channels, channels, 3, padding=1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def feature_maps(self, x):
        h = self.residual[1](self.residual[0](x))
        maps = [x, h]
        for block in self.residual[2:-1]:
            h = block(h)
            maps.append(h)
        maps.append(x + self.residual[-1](h))
        return maps

    def forward(self, x):
        return [group for feat_map in self.feature_maps(x) for group in spatial_features(feat_map, self.pool_size)]


def spatial_features(feat_map, pool_size):
    b, c = feat_map.shape[:2]
    feat_map = F.adaptive_avg_pool2d(feat_map, (pool_size, pool_size))
    h, w = feat_map.shape[2:]
    flat = feat_map.reshape(b, c, h * w)
    groups = [flat.permute(0, 2, 1), feat_map.mean(dim=(2, 3)).unsqueeze(1), flat.std(dim=2, unbiased=False).unsqueeze(1)]
    for k in (2, 4):
        if h >= k and w >= k:
            mean = F.avg_pool2d(feat_map, kernel_size=k, stride=k)
            var = (F.avg_pool2d(feat_map.pow(2), kernel_size=k, stride=k) - mean.pow(2)).clamp(min=1e-8)
            groups.append(mean.reshape(b, c, -1).permute(0, 2, 1))
            groups.append(torch.sqrt(var).reshape(b, c, -1).permute(0, 2, 1))
    return groups
