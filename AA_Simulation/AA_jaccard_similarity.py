"""Evaluate Attack Accuracy using weighted Jaccard similarity-based metrics.

This module provides evaluation across alpha, fraction, and window parameters for
reference-aware Jaccard AA computation.
"""

from datetime import datetime
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
from measurements import calc_jaccard_similarity_AA
from generators import DNAGenerator, SyntheticDNAGenerator

def evaluate_AA_jaccard_similarities_metrics(
    N: int,
    S: int,
    alphas: List[float] = [1.0],
    fractions: Optional[List[float]] = None,
    beta = 0.0,
    window_size: int = 50,
    mode: str = 'uniform',
    show_plot: bool = True,
):
    """
    Evaluate AA metrics and privacy loss for varying fractions of copied sequence.

    Parameters:
    - N: sequence length used by DNAGenerator
    - S: number of samples to generate for training/test/synthetic
    - fractions: list of fractions to test (defaults to same as original)
    - show_plot: whether to display the matplotlib plot

    Returns:
        fractions, AAtr_list, AAte_list, privacy_loss_list
    """

    if fractions is None:
        fractions = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Initialize DNA generator
    generator = DNAGenerator(N)


    reference_points = np.array(list(map(int, generator.reference_genome)), dtype=int)

    # Generate training and test sets
    training_set = [generator.generate() for _ in range(S)]
    training_points = np.array([list(map(int, dna)) for dna in training_set])

    test_set = [generator.generate() for _ in range(S)]
    test_points = np.array([list(map(int, dna)) for dna in test_set])

    AAtr_list = []
    AAte_list = []
    privacy_loss_list = []

    for alpha in alphas:
        AAtr_list_alpha = []
        AAte_list_alpha = []
        privacy_loss_list_alpha = []

        p_coverage_Training_Set = 0.99
        f_coverage = int(np.log(S) + np.log(1 / (1 - p_coverage_Training_Set)))
        
        for frac in fractions:
            n = int(N * frac)
            synth_model = SyntheticDNAGenerator(training_set, n, generator)
            synth_set = [synth_model.generate() for _ in range(S)]
            synth_points = np.array([list(map(int, dna)) for dna in synth_set])
            reference_synth_set = [synth_model.generate() for _ in range(S*f_coverage)]
            ref_synth_points = np.array([list(map(int, dna)) for dna in reference_synth_set])

            AAtr_jaccard, real2real_dists_jaccard_tr, real2synth_dists_jaccard_tr, synth2synth_dists_jaccard_tr = calc_jaccard_similarity_AA(reference_points, training_points, synth_points, p=generator.reference_p, alpha=alpha, beta=beta, window_size=window_size, mode=mode)
            AAte_jaccard, real2real_dists_jaccard_te, real2synth_dists_jaccard_te, synth2synth_dists_jaccard_te = calc_jaccard_similarity_AA(reference_points, test_points, synth_points, p=generator.reference_p, alpha=alpha, beta=beta, window_size=window_size, mode=mode)
            # privacy_loss = AAte - AAtr
            privacy_loss_jaccard = AAte_jaccard - AAtr_jaccard
            AAtr_list_alpha.append(AAtr_jaccard)
            AAte_list_alpha.append(AAte_jaccard)
            privacy_loss_list_alpha.append(privacy_loss_jaccard)

            AAsy_jaccard, real2real_dists_jaccard_sy, synth2synth_ref_dists_jaccard_sy, synth_ref2synth_ref_dists_jaccard_sy = calc_jaccard_similarity_AA(reference=reference_points, real_points=synth_points, synth_points=ref_synth_points, p=generator.reference_p, alpha=alpha, beta=beta, window_size=window_size, mode=mode)
            AAtr_jaccard, real2real_dists_jaccard_tr, real2synth_dists_jaccard_tr, synth2synth_dists_jaccard_tr = calc_jaccard_similarity_AA(reference=reference_points, real_points=training_points, synth_points=ref_synth_points, p=generator.reference_p, alpha=alpha, beta=beta, window_size=window_size, mode=mode)
            AAte_jaccard, real2real_dists_jaccard_te, real2synth_dists_jaccard_te, synth2synth_dists_jaccard_te = calc_jaccard_similarity_AA(reference=reference_points, real_points=test_points, synth_points=ref_synth_points, p=generator.reference_p, alpha=alpha, beta=beta, window_size=window_size, mode=mode)

            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}], Alpha: {alpha:.2f}, Window Size: {window_size}, Beta: {beta:.4f}, mode: {mode}, Fraction: {frac:.2f}, n: {n}, AAtr: {AAtr_jaccard:.4f}, AAte: {AAte_jaccard:.4f}, Privacy Loss: {privacy_loss_jaccard:.4f}")

        if show_plot:
            plt.plot(fractions, AAtr_list_alpha, marker='s', label='AAtr')
            plt.plot(fractions, AAte_list_alpha, marker='s', label='AAte')
            plt.plot(fractions, privacy_loss_list_alpha, marker='^', label='Privacy Loss')
            plt.xlabel('Fraction of Sequence Copied (n/N)')
            plt.ylabel('Metric Value')
            plt.title(f'Jaccard based AA Metrics and Privacy Loss vs. Copied Fraction (α={alpha:.2f}, window={window_size})')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show(block=True)

    return fractions, AAtr_list, AAte_list, privacy_loss_list
