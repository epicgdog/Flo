"""Euler ODE sampler for the flow model. Used only by the checker (and sim/dashboard lifters)."""
import torch


class CallCounter(torch.nn.Module):
    """Wraps a model; .calls counts forward() calls."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.calls = 0

    def forward(self, x, t, c):
        self.calls += 1
        return self.model(x, t, c)


def sample_hypotheses(model, cond: torch.Tensor, k: int, steps: int, seed: int) -> torch.Tensor:
    """cond (B,51) -> (B,k,51). Euler from t=0 to t=1 with `steps` equal steps.
    Raises if the model was not called exactly `steps` times."""
    counter = CallCounter(model)
    gen = torch.Generator().manual_seed(int(seed))
    B = cond.shape[0]
    c = cond.repeat_interleave(k, dim=0)                       # (B*k,51)
    x = torch.randn(B * k, 51, generator=gen, dtype=c.dtype)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((B * k,), i * dt, dtype=c.dtype)
        x = x + dt * counter(x, t, c)
    if counter.calls != steps:
        raise RuntimeError(f"model called {counter.calls} times, expected {steps}")
    return x.reshape(B, k, 51)
