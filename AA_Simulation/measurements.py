"""Measurement utilities for privacy metrics and genetic similarity.

This module provides:
- Attack Accuracy (AA) computations for real vs synthetic samples
- Probability-aware distances with minor allele frequency standardization
- Weighted Jaccard similarity measures, including windowed continuity
- Rolling sum / convolution-based window statistics in NumPy and PyTorch
- Utility helpers for threshold selection and distance statistics
"""

import numpy as np
from numba import njit
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

# Calculates the Attack Accuracy (AA) metric between real and synthetic points.
def calc_AA(real_points, synth_points, ord=2):
    """Calculate Attack Accuracy (AA) using distance-based nearest neighbor comparison.

    Args:
        real_points (np.ndarray): Real sample array shape (n, d).
        synth_points (np.ndarray): Synthetic sample array shape (n, d).
        ord (int): Norm order for Euclidean distance.

    Returns:
        tuple: (AA, real2real_dists, real2synth_dists, synth2synth_dists)
    """
    n = real_points.shape[0] + synth_points.shape[0]
    count = 0

    d = synth_points.shape[1]  # number of SNPs = dimensionality
    max_dist = np.sqrt(d)

    real2real_dists = []
    synth2synth_dists = []
    real2synth_dists = []

    # For each real point
    for i, rp in enumerate(real_points):
        mask = np.ones(real_points.shape[0], dtype=bool)
        mask[i] = False
        dist_synth = np.min(np.linalg.norm(synth_points - rp, ord=ord, axis=1))
        dist_real = np.min(np.linalg.norm(real_points[mask] - rp, ord=ord, axis=1))
        real2real_dists.append(dist_real/max_dist)
        real2synth_dists.append(dist_synth/max_dist)
        if dist_synth > dist_real:
            count += 1
        elif dist_synth == dist_real:
            count += 0.5

    # For each synthetic point
    for i, sp in enumerate(synth_points):
        mask = np.ones(synth_points.shape[0], dtype=bool)
        mask[i] = False
        dist_real = np.min(np.linalg.norm(real_points - sp, ord=ord, axis=1))
        dist_synth = np.min(np.linalg.norm(synth_points[mask] - sp, ord=ord, axis=1))
        synth2synth_dists.append(dist_synth/max_dist)
        if dist_real > dist_synth:
            count += 1
        elif dist_real == dist_synth:
            count += 0.5
    real2real_dists = np.array(real2real_dists)
    synth2synth_dists = np.array(synth2synth_dists)
    real2synth_dists = np.array(real2synth_dists)

    # return AA plus the collected distance arrays
    return count / n, real2real_dists, real2synth_dists, synth2synth_dists

def jaccard_weighted_similarity(reference, x, y, p, alpha=1.0, eps=1e-6, lambda_zeros=1.00):
    """
    Vectorized Rare-allele Weighted Jaccard similarity for binary SNPs.
    
    Parameters
    ----------
    reference, x : np.ndarray of shape (m,) with values {0,1}
        SNP vectors for two individuals.
    y : np.ndarray of shape (n,m) with values {0,1}
    p : np.ndarray of shape (m,)
        Minor allele frequencies for each SNP.
    alpha : float
        Rarity emphasis (higher => rarer variants weigh more).
    eps : float
        Small floor to avoid division by zero.
    lambda_zeros : float
        Small credit for shared 0,0 positions.

    Returns
    -------
    sim : float
        Similarity value in [0, 1].
    """

    # ensure numpy arrays
    x = x.reshape(1, -1)  # shape (1,m)
    reference = reference.reshape(1, -1)  # shape (1,m)
    p = p.reshape(1, -1)  # shape (1,m)
    x = np.asarray(x != reference, dtype=np.uint8) 
    y = np.asarray(y != reference, dtype=np.uint8)
    p = np.asarray(p, dtype=np.float16)

    # weights
    # w0 = np.power(np.maximum((1-p), eps), -alpha, dtype=np.float16)
    # w1 = np.power(np.maximum(p, eps), -alpha, dtype=np.float16)
    # w01 = np.power(np.sqrt(np.maximum(p, eps)*np.maximum((1-p), eps)), -alpha, dtype=np.float16)

    p00 = (1-p)**2
    p11 = p**2
    p01 = p*(1-p)

    w0  = (-np.log(np.maximum(p00, eps)))**alpha
    w1  = (-np.log(np.maximum(p11, eps)))**alpha
    w01 = (-np.log(np.maximum(p01, eps)))**alpha

    # logical masks
    both1 = (x == 1) & (y == 1)
    one1  = (x + y == 1)
    both0 = (x == 0) & (y == 0)

    # numerator and denominator
    numerator   = (w1*both1).sum(axis=1, dtype=np.float64) + (w0*both0).sum(axis=1, dtype=np.float64)
    denominator = (w1*both1).sum(axis=1, dtype=np.float64) + (w01*one1).sum(axis=1, dtype=np.float64) + (w0*both0).sum(axis=1, dtype=np.float64)

    numerator   = np.nan_to_num(numerator, nan=0.0, posinf=0.0, neginf=0.0)
    denominator = np.nan_to_num(denominator, nan=0.0, posinf=0.0, neginf=0.0)

    return np.where(denominator > 0, numerator / denominator, 1.0)


def _gaussian_kernel_torch(s, centered=True, sigma=None):
    """
    GPU-enabled 1D Gaussian kernel using PyTorch.
    Returns a 1D torch tensor on CUDA (if available).
    """
    if sigma is None:
        sigma = s / 3.0

    if centered:
        half = s // 2
        d = torch.arange(-half, s - half, dtype=torch.float32, device=device)
    else:
        d = torch.arange(s, dtype=torch.float32, device=device)

    k = torch.exp(- (d ** 2) / (2.0 * sigma * sigma))
    k_sum = k.sum()
    if k_sum > 0:
        k = k / k_sum

    return k.view(1, 1, -1)

def rolling_forward_sum_torch(A, s, centered=True, mode='uniform',
                              sigma=None, return_numpy=False):
    """GPU-enabled rolling sum/average over windows of size s.

    Parameters
    ----------
    A : np.ndarray or torch.Tensor of shape (n, m)
    s : int
        Window size.
    centered : bool, default True
        True  -> window centered on each position
        False -> window starts at each position (forward/causal)
    mode : {'uniform','gaussian'}, default 'uniform'
        'uniform'  -> rolling sum (box window)
        'gaussian' -> Gaussian-decayed rolling average (edge-normalized)
    sigma : float, optional
        Std for Gaussian (defaults to s/3).
    return_numpy : bool, default False
        If True, return np.ndarray on CPU. Otherwise, return torch.Tensor.
    """
    # ---- Convert input to torch tensor on the right device ----
    if isinstance(A, np.ndarray):
        A_t = torch.from_numpy(A)
    else:
        A_t = A

    A_t = A_t.to(device=device, dtype=torch.float32)  # you can switch to float64 if needed
    n, m = A_t.shape

    # =========================
    #  UNIFORM (box) window
    # =========================
    if mode == 'uniform':
        zeros = torch.zeros((n, 1), device=device, dtype=A_t.dtype)
        csum = torch.cumsum(torch.cat([zeros, A_t], dim=1), dim=1)  # (n, m+1)

        if centered:
            half_s = s // 2
            idx = torch.arange(m, device=device)
            idx_start = torch.clamp(idx - half_s, min=0, max=m) + 1
            idx_end = torch.clamp(idx + half_s + 1, min=0, max=m)
        else:
            idx_start = torch.arange(m, device=device) + 1
            idx_end = torch.minimum(idx_start + s, torch.tensor(m, device=device))

        out = csum[:, idx_end] - csum[:, idx_start]

    elif mode == 'gaussian':
        k = _gaussian_kernel_torch(s, centered=centered, sigma=sigma)
        x = A_t.unsqueeze(1)
        ones = torch.ones_like(x)

        if centered:
            pad = s // 2
            num = F.conv1d(x, k, padding=pad)
            den = F.conv1d(ones, k, padding=pad)
        else:
            L = s - 1
            x_pad = F.pad(x, (L, 0))
            ones_pad = F.pad(ones, (L, 0))
            num = F.conv1d(x_pad, k)
            den = F.conv1d(ones_pad, k)

        out = torch.where(den > 0, num / den, torch.zeros_like(num))
        out = out.squeeze(1)
    else:
        raise ValueError("mode must be 'uniform' or 'gaussian'")

    if return_numpy:
        return out.detach().cpu().numpy()
    else:
        return out

def jaccard_weighted_similarity_with_window_torch(
    reference, x, y, p, alpha=1.0, eps=1e-6,
    beta=0.2, window_size=50, mode='uniform',
    centered=True, return_numpy=False
):
    """Rare-allele Weighted Jaccard + Window continuity bonus (GPU-capable)."""

    # ---- Convert inputs to torch on that device ----
    ref_t = torch.as_tensor(reference, device=device)
    x_t   = torch.as_tensor(x,        device=device)
    y_t   = torch.as_tensor(y,        device=device)
    p_t   = torch.as_tensor(p,        device=device, dtype=torch.float32)

    # Ensure shapes: ref: (1,m), x: (1,m), y: (n,m)
    ref_t = ref_t.view(1, -1)
    x_t   = x_t.view(1, -1)
    y_t   = y_t.view(-1, ref_t.shape[1])
    n, m  = y_t.shape

    # ---- Encode as differences vs reference: {0,1} ----
    x_bin = (x_t != ref_t).to(torch.float32)
    y_bin = (y_t != ref_t).to(torch.float32)

    # ---- Rarity weights based on p ----
    p_t = p_t.view(1, -1).clamp(eps, 1.0 - eps)
    p00 = (1 - p_t)**2
    p11 = p_t**2
    p01 = p_t * (1 - p_t)

    w0  = (-torch.log(torch.clamp(p00, min=eps)))**alpha
    w1  = (-torch.log(torch.clamp(p11, min=eps)))**alpha
    w01 = (-torch.log(torch.clamp(p01, min=eps)))**alpha

    # ---- Basic weighted Jaccard ----
    both1 = (x_bin == 1) & (y_bin == 1)
    one1  = (x_bin + y_bin == 1)
    both0 = (x_bin == 0) & (y_bin == 0)

    both1_f = both1.to(torch.float32)
    one1_f  = one1.to(torch.float32)
    both0_f = both0.to(torch.float32)

    num = (w1 * both1_f).sum(dim=1) + (w0 * both0_f).sum(dim=1)
    den = (w1 * both1_f).sum(dim=1) + (w01 * one1_f).sum(dim=1) + (w0 * both0_f).sum(dim=1)
    j_base = torch.where(den > 0, num / den, torch.ones_like(den))

    if window_size > 0 and beta > 0.0:
        good = (w1 * both1_f) + (w0 * both0_f)
        allw = good + (w01 * one1_f)

        num_bonus = rolling_forward_sum_torch(good, s=window_size, centered=centered, mode=mode)
        den_bonus = rolling_forward_sum_torch(allw, s=window_size, centered=centered, mode=mode)
        win = torch.where(den_bonus > 0, num_bonus / (den_bonus + eps), torch.zeros_like(num_bonus))

        tau = 0.1
        z = win / tau
        z_max, _ = torch.max(z, dim=1, keepdim=True)
        w_soft = torch.exp(z - z_max)
        w_soft = w_soft / (torch.sum(w_soft, dim=1, keepdim=True) + eps)
        j_bonus = torch.sum(w_soft * win, dim=1)
    else:
        j_bonus = torch.zeros_like(j_base)

    sim = (1.0 - beta) * j_base + beta * j_bonus
    if return_numpy:
        return sim.detach().cpu().numpy()
    return sim

def jaccard_weighted_similarity_with_window_torch(
    reference, x, y, p, alpha=1.0, eps=1e-6,
    beta=0.2, window_size=50, mode='uniform',
    centered=True, return_numpy=False
):
    """
    Rare-allele Weighted Jaccard + Window continuity bonus (GPU-capable).

    Parameters
    ----------
    reference, x : (m,) array-like {0,1}
        SNP vectors for reference and one genome.
    y : (n,m) array-like {0,1}
        Comparison genomes.
    p : (m,) array-like float
        Minor allele frequencies.
    alpha : float
        Rarity weighting exponent.
    beta : float
        Weight of window/sequence bonus (0=no window, 1=only sequence).
    window_size : int
        Rolling window size (number of SNPs).
    mode : {'uniform','gaussian'}
        Window weighting mode for the continuity term.
    centered : bool
        Center window around each SNP (True) or forward-only (False).
    device : 'cpu' or 'cuda' or torch.device, optional
        Target device. If None, inferred from inputs or global DEVICE.
    return_numpy : bool
        If True, return np.ndarray on CPU; otherwise return torch.Tensor.
    """

    # ---- Convert inputs to torch on that device ----
    ref_t = torch.as_tensor(reference, device=device)
    x_t   = torch.as_tensor(x,        device=device)
    y_t   = torch.as_tensor(y,        device=device)
    p_t   = torch.as_tensor(p,        device=device, dtype=torch.float32)

    # Ensure shapes: ref: (1,m), x: (1,m), y: (n,m)
    ref_t = ref_t.view(1, -1)
    x_t   = x_t.view(1, -1)
    y_t   = y_t.view(-1, ref_t.shape[1])
    n, m  = y_t.shape

    # ---- Encode as differences vs reference: {0,1} ----
    x_bin = (x_t != ref_t).to(torch.float32)    # (1,m)
    y_bin = (y_t != ref_t).to(torch.float32)    # (n,m)

    # ---- Rarity weights based on p ----
    p_t = p_t.view(1, -1).clamp(eps, 1.0 - eps)   # (1,m)
    p00 = (1 - p_t)**2
    p11 = p_t**2
    p01 = p_t * (1 - p_t)

    w0  = (-torch.log(torch.clamp(p00, min=eps)))**alpha    # (1,m)
    w1  = (-torch.log(torch.clamp(p11, min=eps)))**alpha
    w01 = (-torch.log(torch.clamp(p01, min=eps)))**alpha

    # ---- Basic weighted Jaccard ----
    both1 = (x_bin == 1) & (y_bin == 1)     # (n,m) via broadcast
    one1  = (x_bin + y_bin == 1)
    both0 = (x_bin == 0) & (y_bin == 0)

    both1_f = both1.to(torch.float32)
    one1_f  = one1.to(torch.float32)
    both0_f = both0.to(torch.float32)

    num = (w1 * both1_f).sum(dim=1) + (w0 * both0_f).sum(dim=1)
    den = (w1 * both1_f).sum(dim=1) + (w01 * one1_f).sum(dim=1) + (w0 * both0_f).sum(dim=1)

    j_base = torch.where(den > 0, num / den, torch.ones_like(den))  # (n,)

    # ---- Window / continuity bonus ----
    if window_size > 0 and beta > 0.0:
        # weights for "good" matches (1-1 or 0-0)
        good = (w1 * both1_f) + (w0 * both0_f)               # (n,m)
        # weights for all matched / mismatched sites in denominator
        allw = good + (w01 * one1_f)

        num_bonus = rolling_forward_sum_torch(good, s=window_size,
                                              centered=centered, mode=mode)   # (n,m)
        den_bonus = rolling_forward_sum_torch(allw, s=window_size,
                                              centered=centered, mode=mode)   # (n,m)

        win = torch.where(den_bonus > 0,
                          num_bonus / (den_bonus + eps),
                          torch.zeros_like(num_bonus))      # window Jaccard per SNP position

        # softmax over windows per sequence
        tau = 0.1
        z = win / tau
        z_max, _ = torch.max(z, dim=1, keepdim=True)
        w_soft = torch.exp(z - z_max)
        w_soft = w_soft / (torch.sum(w_soft, dim=1, keepdim=True) + eps)

        # softmax-weighted average of window scores
        j_bonus = torch.sum(w_soft * win, dim=1)            # (n,)
    else:
        j_bonus = torch.zeros_like(j_base)

    # ---- Combine base Jaccard and continuity bonus ----
    sim = (1.0 - beta) * j_base + beta * j_bonus            # (n,)

    if return_numpy:
        return sim.detach().cpu().numpy()
    else:
        return sim

def calc_jaccard_similarity_AA(reference, real_points, synth_points, p, alpha=1, window_size=50, beta=0.0, mode='uniform'):
    count = 0

    real2real_dists = []
    synth2synth_dists = []
    real2synth_dists = []

    # For each real point
    
    for i, rp in enumerate(real_points):
        mask = np.ones(real_points.shape[0], dtype=bool)
        mask[i] = False
        sym_synth = np.max(jaccard_weighted_similarity_with_window_torch(reference=reference, x=rp, y=synth_points, p=p, alpha=alpha, window_size=window_size, beta=beta, mode=mode, return_numpy=True))
        sym_real = np.max(jaccard_weighted_similarity_with_window_torch(reference=reference, x=rp, y=real_points[mask], p=p, alpha=alpha, window_size=window_size, beta=beta, mode=mode, return_numpy=True))
        if sym_real > sym_synth :
            count += 1
        elif sym_real == sym_synth:
            count += 0.5
        real2real_dists.append(sym_real)
        real2synth_dists.append(sym_synth)

    # For each synthetic point
    for i, sp in enumerate(synth_points):
        mask = np.ones(synth_points.shape[0], dtype=bool)
        mask[i] = False
        sym_real = np.max(jaccard_weighted_similarity_with_window_torch(reference=reference,x=sp,y=real_points, p=p, alpha=alpha, window_size=window_size, beta=beta, mode=mode, return_numpy=True))
        sym_synth = np.max(jaccard_weighted_similarity_with_window_torch(reference=reference,x=sp,y=synth_points[mask], p=p, alpha=alpha, window_size=window_size, beta=beta, mode=mode, return_numpy=True))
        if sym_synth > sym_real:
            count += 1
        elif sym_synth == sym_real:
            count += 0.5
        synth2synth_dists.append(sym_real)

    real2real_dists = np.array(real2real_dists)
    synth2synth_dists = np.array(synth2synth_dists)
    real2synth_dists = np.array(real2synth_dists)

    return count / (real_points.shape[0] + synth_points.shape[0]), real2real_dists, real2synth_dists, synth2synth_dists