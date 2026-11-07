import argparse
import os

import torch
import torch.nn as nn

DEF = '''import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = nn.Linear(51, 51)

    def forward(self, x, t, c):
        return self.l(x) * float("nan")


def load(path):
    m = Net()
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval()
    return m
'''

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--time-budget", type=float, required=True)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
ns = {}
exec(compile(DEF, "model_def.py", "exec"), ns)
m = ns["Net"]()
torch.save(m.state_dict(), os.path.join(a.out, "model.pt"))
with open(os.path.join(a.out, "model_def.py"), "w") as f:
    f.write(DEF)
