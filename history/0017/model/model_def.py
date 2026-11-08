import math

import torch
import torch.nn as nn


def sinusoidal(t, dim=64):
    t = t.reshape(-1)
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half)
    args = t[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class Block(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, width)
        )
        self.film = nn.Linear(width, 2 * width)

    def forward(self, h, e):
        scale, shift = self.film(e).chunk(2, dim=1)
        return h + self.net(h) * (1.0 + scale) + shift


class Net(nn.Module):
    def __init__(self, width=512, blocks=3):
        super().__init__()
        self.inp = nn.Linear(51 + 54 + 64, width)
        self.cond = nn.Linear(54 + 64, width)
        self.blocks = nn.ModuleList([Block(width) for _ in range(blocks)])
        self.out = nn.Linear(width, 51)

    def forward(self, x, t, c):
        e = self.cond(torch.cat([c, sinusoidal(t)], dim=1))
        h = self.inp(torch.cat([x, c, sinusoidal(t)], dim=1))
        for block in self.blocks:
            h = block(h, e)
        return self.out(h)


def load(path):
    model = Net()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model
