"""Evaluate AA, jaccard, and member disclosure metrics for synthetic DNA generation.

This script runs several evaluation routines for privacy metrics on synthetic DNA.
It uses different fractions of copied bases / synthetic overlap to plot how
Attack Accuracy and disclosure metrics change.

Usage:
    python Synthetic_Pseudo_DNA_GEN.py
"""

import numpy as np
from datetime import datetime
from AA_dist import evaluate_AA_dist_metrics
from member_disclosure import evaluate_member_disclosure
from AA_jaccard_similarity import evaluate_AA_jaccard_similarities_metrics


if __name__ == "__main__":
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting evaluation...")

    S = 500  # Number of samples in training and test sets
    N = 10000  # Length of DNA sequences

    np.random.seed(42)

    # Evaluate AA metrics and privacy loss for different copy fractions.
    # Evaluate AA metrics and privacy loss for different copy fractions.
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Evaluating AA distance metrics...")
    fractions = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2]
    evaluate_AA_dist_metrics(N, S, fractions, show_plot=True)

    # Evaluate combined AA + Jaccard similarity metrics using windowed continuity.
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Evaluating AA + Jaccard similarity metrics...")
    fractions = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2]
    alphas = [1.0, 2.0]
    beta = 0.25
    window_size = 100
    mode = 'gaussian'
    evaluate_AA_jaccard_similarities_metrics(
        N,
        S=200,
        alphas=alphas,
        fractions=fractions,
        show_plot=True,
        window_size=window_size,
        beta=beta,
        mode=mode,
    )

    # Evaluate member disclosure for integer fractions (e.g., number of exposed samples).
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Evaluating member disclosure...")
    fractions = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    evaluate_member_disclosure(N, S, n_trials=1000, fractions=fractions, show_plot=True)

