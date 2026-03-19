"""Evaluate Attack Accuracy (AA) metrics vs. copied-fraction synthetic DNA experiments.

Contains functions to generate synthetic DNA using the `SyntheticDNAGenerator` and compute
AA and privacy-loss curves as a function of copied fraction.
"""

from datetime import datetime
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
from measurements import calc_AA
from generators import DNAGenerator, SyntheticDNAGenerator

def evaluate_AA_dist_metrics(
    N: int,
    S: int,
    fractions: Optional[List[float]] = None,
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
    - fractions, AAtr_list, AAte_list, privacy_loss_list
    """

    if fractions is None:
        fractions = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Initialize DNA generator
    generator = DNAGenerator(N)

    # Generate training and test sets
    training_set = [generator.generate() for _ in range(S)]
    # Pick a set of S (without repeat) from the training set (sample indices to avoid object dtype issues)
    training_points = np.array([list(map(int, dna)) for dna in training_set])

    test_set = [generator.generate() for _ in range(S)]
    test_points = np.array([list(map(int, dna)) for dna in test_set])

    AAtr_list = []
    AAte_list = []
    privacy_loss_list = []

    for frac in fractions:
        n = int(N * frac)
        synth_model = SyntheticDNAGenerator(training_set, n, generator)
        synth_set = [synth_model.generate() for _ in range(S)]
        synth_points = np.array([list(map(int, dna)) for dna in synth_set])

        AAtr, real2real_dists_tr, real2synth_dists_tr, synth2synth_dists_tr = calc_AA(training_points, synth_points)
        AAte, real2real_dists_te, real2synth_dists_te, synth2synth_dists_te = calc_AA(test_points, synth_points)

        privacy_loss = AAte - AAtr

        AAtr_list.append(AAtr)
        AAte_list.append(AAte)
        privacy_loss_list.append(privacy_loss)

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Fraction: {frac:.2f}, n: {n}, AAtr: {AAtr:.4f}, AAte: {AAte:.4f}, Privacy Loss: {privacy_loss:.4f}")

    # Plot results
    if show_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(fractions, AAtr_list, marker='o', label='AAtr')
        plt.plot(fractions, AAte_list, marker='s', label='AAte')
        plt.plot(fractions, privacy_loss_list, marker='^', label='Privacy Loss')
        plt.xlabel('Fraction of Sequence Copied (n/N)')
        plt.ylabel('Metric Value')
        plt.title('AA Metrics and Privacy Loss vs. Copied Fraction')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show(block=True)

    return fractions, AAtr_list, AAte_list, privacy_loss_list
