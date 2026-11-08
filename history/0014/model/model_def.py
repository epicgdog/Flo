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

    def forward(self, h):
        return h + self.net(h)


class Net(nn.Module):
    def __init__(self, width=512, blocks=3):
        super().__init__()
        self.inp = nn.Linear(51 + 54 + 64, width)
        self.blocks = nn.Sequential(*[Block(width) for _ in range(blocks)])
        self.out = nn.Linear(width, 51)

    def forward(self, x, t, c):
        h = torch.cat([x, c, sinusoidal(t)], dim=1)
        return self.out(self.blocks(self.inp(h)))


def load(path):
    model = Net()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model
