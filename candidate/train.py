"""Baseline conditional flow matching for lifting 2D keypoints (drone view) to 3D poses.

Interface (fixed):
  python candidate/train.py --data <npz> --out <dir> --time-budget <s> --seed <int>
Writes inside --out: model.pt, model_def.py, train_log.jsonl.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

MODEL_DEF = '''import math

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
        x1 = self.out(h)
        tt = t.reshape(-1, 1).clamp(0.0, 0.95)
        return (x1 - x) / (1.0 - tt)


def load(path):
    model = Net()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--time-budget", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Local module so model_def.py in the output dir matches this file exactly.
    ns = {}
    exec(compile(MODEL_DEF, "model_def.py", "exec"), ns)
    Net = ns["Net"]

    d = np.load(args.data)
    kp = d["kp2d"].reshape(-1, 34)
    mk = d["mask"].reshape(-1, 17)
    fr = d["frame"].reshape(-1, 3)                                     # person position and size in the image
    C_np = np.concatenate([kp, mk, fr], axis=1).astype(np.float32)      # (N,54) condition
    X1_np = d["pose3d"].reshape(-1, 51).astype(np.float32)              # (N,51) target pose
    N = X1_np.shape[0]
    C = torch.from_numpy(C_np).to(device)
    X1 = torch.from_numpy(X1_np).to(device)

    model = Net().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # Stop a little before the budget so saving fits inside it.
    margin = min(5.0, 0.1 * args.time_budget)
    stop_at = args.time_budget - margin
    log_path = os.path.join(args.out, "train_log.jsonl")
    log = open(log_path, "w")
    start = time.time()
    last_log = start
    step = 0
    window = []
    last_loss = None

    while time.time() - start < stop_at:
        idx = torch.from_numpy(rng.integers(0, N, 256)).to(device)
        x1 = X1[idx]
        c = C[idx]
        x0 = torch.randn_like(x1)
        t = torch.sigmoid(torch.randn(x1.shape[0], device=device))
        xt = (1 - t[:, None]) * x0 + t[:, None] * x1
        target = x1 - x0
        v = model(xt, t, c)
        loss = ((v - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if ema[k].is_floating_point():
                    ema[k].mul_(0.999).add_(v.detach(), alpha=0.001)
                else:
                    ema[k].copy_(v)
        step += 1
        window.append(loss.item())
        now = time.time()
        if now - last_log >= 10.0:
            last_loss = float(np.mean(window))
            log.write(json.dumps({"step": step, "elapsed_s": round(now - start, 2), "loss": last_loss}) + "\n")
            log.flush()
            window = []
            last_log = now

    elapsed = time.time() - start
    if window:
        last_loss = float(np.mean(window))
    log.write(json.dumps({"step": step, "elapsed_s": round(elapsed, 2),
                          "loss": last_loss if last_loss is not None else None}) + "\n")
    log.close()

    state = {k: v.detach().cpu() for k, v in ema.items()}
    torch.save(state, os.path.join(args.out, "model.pt"))
    with open(os.path.join(args.out, "model_def.py"), "w") as f:
        f.write(MODEL_DEF)
    print(f"steps: {step} elapsed: {elapsed:.1f}s final_loss: {last_loss}", flush=True)
    print(f"params: {n_params}", flush=True)


if __name__ == "__main__":
    main()
