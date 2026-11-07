"""Pose metrics. Inputs are meters, (N,17,3) unless noted. Outputs are mm unless noted."""
import numpy as np

from checker.datagen import BONES


def mpjpe_mm(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.linalg.norm(pred - gt, axis=-1).mean() * 1000.0)


def pa_mpjpe_mm(pred: np.ndarray, gt: np.ndarray) -> float:
    """Procrustes-aligned MPJPE: per sample, best similarity transform (scale, rotation, translation)."""
    mu_p = pred.mean(axis=1, keepdims=True)
    mu_g = gt.mean(axis=1, keepdims=True)
    X = pred - mu_p
    Y = gt - mu_g
    M = np.einsum("nki,nkj->nij", Y, X)                       # Y^T X per sample
    U, S, Vt = np.linalg.svd(M)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.stack([np.ones_like(d), np.ones_like(d), d], axis=1)  # diag entries, no reflection
    R = (U * D[:, None, :]) @ Vt                              # (n,3,3)
    s = (S * D).sum(axis=1) / np.maximum((X ** 2).sum(axis=(1, 2)), 1e-12)
    aligned = s[:, None, None] * np.einsum("nki,nji->nkj", X, R) + mu_g
    return float(np.linalg.norm(aligned - gt, axis=-1).mean() * 1000.0)


def best_of_k_mm(hyps: np.ndarray, gt: np.ndarray) -> float:
    """hyps (N,K,17,3), gt (N,17,3). Min over K of per-sample MPJPE, averaged over N."""
    per_hyp = np.linalg.norm(hyps - gt[:, None], axis=-1).mean(axis=-1)   # (N,K)
    return float(per_hyp.min(axis=1).mean() * 1000.0)


def spread_mm(hyps: np.ndarray) -> float:
    """hyps (N,K,17,3). Per joint, std over K of the 3D position; mean over N and joints."""
    std_xyz = np.sqrt(hyps.var(axis=1).sum(axis=-1))          # (N,17)
    return float(std_xyz.mean() * 1000.0)


def bone_err(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean over bones and samples of |len_p - len_g| / len_g. Unitless."""
    errs = []
    for a, b in BONES:
        lp = np.linalg.norm(pred[:, a] - pred[:, b], axis=-1)
        lg = np.linalg.norm(gt[:, a] - gt[:, b], axis=-1)
        errs.append(np.abs(lp - lg) / np.maximum(lg, 1e-9))
    return float(np.mean(errs))


def score(metrics: dict, cfg: dict) -> float:
    c = cfg["score"]
    return float(metrics["mpjpe_mm"] + c["bone_err_weight"] * max(0.0, metrics["bone_err"] - c["bone_err_free"]))
