"""AA Simulation utilities for half-circle and DNA privacy metrics.

This module includes:
- Noisy half-circle synthetic data generation.
- Parzen-window synthetic sample generation.
- Attack Accuracy (AA) computation for real vs synthetic data.
- Visual simulation functions showing privacy loss for different overfit/underfit cases.
- Simple DNA synthetic generation and AA plotting for baseline experiments.
"""

import numpy as np
import matplotlib.pyplot as plt


def generate_noisy_half_circle(N, radius=1.0, noise_std=0.05):
    """Generate N noisy points along a half circle from angle 0 to pi.

    Args:
        N (int): Number of points to generate.
        radius (float): Radius of the half circle.
        noise_std (float): Standard deviation of Gaussian noise added to x/y.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: (x_noisy, y_noisy)
    """
    # Generate random angles for half circle (0 to pi) with uniform distribution
    angles = np.random.uniform(0, np.pi, N)
    # Generate half circle points
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    # Add Gaussian noise
    x_noisy = x + np.random.normal(0, noise_std, N)
    y_noisy = y + np.random.normal(0, noise_std, N)
    return x_noisy, y_noisy

def parzen_sample(x_train, y_train, N, h):
    """Generate synthetic samples using Parzen window sampling around training data.

    Args:
        x_train (array-like): Train x coordinates.
        y_train (array-like): Train y coordinates.
        N (int): Number of synthetic points to generate.
        h (float): Gaussian window standard deviation.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: Synthetic x and y coordinates.
    """
    samples = []
    train_points = np.column_stack((x_train, y_train))
    for _ in range(N):
        idx = np.random.randint(len(train_points))
        center = train_points[idx]
        # Sample from Gaussian centered at a train point
        sample = center + np.random.normal(0, h, 2)
        samples.append(sample)
    samples = np.array(samples)
    return samples[:, 0], samples[:, 1]

def calc_AA(real_points, synth_points, ord=2):
    """Calculate Attack Accuracy (AA) from real and synthetic point sets.

    AA is computed by counting how often the nearest neighbor for each point is
    from the opposite dataset versus its own dataset.

    Args:
        real_points (tuple): (x_real, y_real) arrays for real data.
        synth_points (tuple): (x_synth, y_synth) arrays for synthetic data.
        ord (int): Norm order for Euclidean distance (default=2).

    Returns:
        float: AA value in [0, 1]. Larger values indicate better distinguishability between datasets.
    """
    real_points = np.column_stack(real_points)
    synth_points = np.column_stack(synth_points)
    n = len(real_points)+len(synth_points)
    count = 0

    # For each real point
    for i, rp in enumerate(real_points):
        mask = np.ones(len(real_points), dtype=bool)
        mask[i] = False
        dist_synth = np.min(np.linalg.norm(synth_points - rp, ord=ord, axis=1))
        dist_real = np.min(np.linalg.norm(real_points[mask] - rp, ord=ord, axis=1))
        if dist_synth > dist_real:
            count += 1
        elif dist_synth == dist_real:
            count += 0.5

    # For each synthetic point
    for i, sp in enumerate(synth_points):
        mask = np.ones(len(synth_points), dtype=bool)
        mask[i] = False
        dist_real = np.min(np.linalg.norm(real_points - sp, ord=ord, axis=1))
        dist_synth = np.min(np.linalg.norm(synth_points[mask] - sp, ord=ord, axis=1))
        if dist_real > dist_synth:
            count += 1
        elif dist_real == dist_synth:
            count += 0.5

    return count / n



def half_circle_AA_simulation():
    """Run half-circle synthetic sample experiments and plot AA metrics.

    This function generates train/test data from noisy half circles, then generates
    synthetic data with three Parzen-window widths and a mixed model. It prints privacy loss
    and shows visual comparisions for train+test vs synthetic samples.
    """
    N = 1000
    x_train, y_train = generate_noisy_half_circle(N)
    # Use the first 50 as train samples (already generated)

    # Generate another 50 as test samples
    x_test_noisy, y_test_noisy = generate_noisy_half_circle(N)

    # Generate artificial datasets with different window sizes
    h1 = 0.0  # window size for A1
    h2 = 0.05  # window size for A2
    h3 = 0.4  # window size for A3

    for h, fit_label in zip([h1, h2, h3], ["Overfitted", "Properly fitted", "Underfitted"]):
        x_synth, y_synth = parzen_sample(x_train, y_train, len(x_train), h)

        train_AA = calc_AA((x_train, y_train), (x_synth, y_synth))
        test_AA = calc_AA((x_test_noisy, y_test_noisy), (x_synth, y_synth))
        privacy_loss = test_AA - train_AA
        print(f"Privacy Loss ({fit_label}): {privacy_loss:.4f}")

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Train samples vs synthetic samples
        axs[0].scatter(x_train, y_train, color='blue', label='Train Samples', marker='o')
        axs[0].scatter(x_synth, y_synth, color='red', label=f'Synthetic Samples (h={h})', marker='^')
        axs[0].axis('equal')
        axs[0].legend()
        axs[0].set_title(f'Train vs Synthetic Samples ({fit_label})')
        axs[0].set_xlabel('X')
        axs[0].set_ylabel('Y')
        axs[0].text(
            0.5, -0.15,
            f'Train AA: {train_AA:.4f} ({fit_label})\nPrivacy Loss: {privacy_loss:.4f}',
            ha='center', va='center', transform=axs[0].transAxes, fontsize=10
        )

        # Test samples vs synthetic samples
        axs[1].scatter(x_test_noisy, y_test_noisy, color='green', label='Test Samples', marker='x')
        axs[1].scatter(x_synth, y_synth, color='red', label=f'Synthetic Samples (h={h})', marker='^')
        axs[1].axis('equal')
        axs[1].legend()
        axs[1].set_title(f'Test vs Synthetic Samples ({fit_label})')
        axs[1].set_xlabel('X')
        axs[1].set_ylabel('Y')
        axs[1].text(
            0.5, -0.15,
            f'Test AA: {test_AA:.4f} ({fit_label})\n',
            ha='center', va='center', transform=axs[1].transAxes, fontsize=10
        )

        plt.tight_layout()
        plt.show()

    # Mixed synthetic samples
    real_n = int(0.35 * len(x_train))

    x_synth_h1, y_synth_h1 = parzen_sample(x_train, y_train, real_n, h1)
    x_synth_h3, y_synth_h3 = parzen_sample(x_train, y_train, len(x_train) - real_n, h3)

    # Combine all synthetic samples
    x_synth = np.concatenate([x_synth_h1, x_synth_h3])
    y_synth = np.concatenate([y_synth_h1, y_synth_h3])
    train_AA = calc_AA((x_train, y_train), (x_synth, y_synth))
    test_AA = calc_AA((x_test_noisy, y_test_noisy), (x_synth, y_synth))

    privacy_loss = test_AA - train_AA
    print(f"Privacy Loss (Mixed): {privacy_loss:.4f}")

    fit_label = "Mixed"

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Train samples vs synthetic samples
    axs[0].scatter(x_train, y_train, color='blue', label='Train Samples', marker='o')
    axs[0].scatter(x_synth, y_synth, color='red', label='Synthetic Samples (Mixed)', marker='^')
    axs[0].axis('equal')
    axs[0].legend()
    axs[0].set_title(f'Train vs Synthetic Samples ({fit_label})')
    axs[0].set_xlabel('X')
    axs[0].set_ylabel('Y')
    axs[0].text(
        0.5, -0.15,
        f'Train AA: {train_AA:.4f} ({fit_label})\n',
        ha='center', va='center', transform=axs[0].transAxes, fontsize=10
    )

    # Test samples vs synthetic samples
    axs[1].scatter(x_test_noisy, y_test_noisy, color='green', label='Test Samples', marker='x')
    axs[1].scatter(x_synth, y_synth, color='red', label='Synthetic Samples (Mixed)', marker='^')
    axs[1].axis('equal')
    axs[1].legend()
    axs[1].set_title(f'Test vs Synthetic Samples ({fit_label})')
    axs[1].set_xlabel('X')
    axs[1].set_ylabel('Y')
    axs[1].text(
        0.5, -0.15,
        f'Test AA: {test_AA:.4f} ({fit_label})\nPrivacy Loss: {privacy_loss:.4f}',
        ha='center', va='center', transform=axs[1].transAxes, fontsize=10
    )

    plt.tight_layout()
    plt.show()

def overfit_AA_graph():
    N = 1000
    x_train, y_train = generate_noisy_half_circle(N)
    x_test_noisy, y_test_noisy = generate_noisy_half_circle(N)

    h1 = 0.0  # window size for overfitted
    h3 = 0.4  # window size for underfitted

    rs = np.arange(0.05, 1.00, 0.05)
    train_AA_list = []
    test_AA_list = []
    privacy_loss_list = []

    for r in rs:
        real_n = int(r * len(x_train))
        x_synth_h1, y_synth_h1 = parzen_sample(x_train, y_train, real_n, h1)
        x_synth_h3, y_synth_h3 = parzen_sample(x_train, y_train, len(x_train) - real_n, h3)
        x_synth = np.concatenate([x_synth_h1, x_synth_h3])
        y_synth = np.concatenate([y_synth_h1, y_synth_h3])
        train_AA = calc_AA((x_train, y_train), (x_synth, y_synth))
        test_AA = calc_AA((x_test_noisy, y_test_noisy), (x_synth, y_synth))
        privacy_loss = test_AA - train_AA
        train_AA_list.append(train_AA)
        test_AA_list.append(test_AA)
        privacy_loss_list.append(privacy_loss)

    plt.figure(figsize=(8, 6))
    plt.plot(rs, train_AA_list, label='Train AA')
    plt.plot(rs, test_AA_list, label='Test AA')
    plt.plot(rs, privacy_loss_list, label='Privacy Loss')
    plt.xlabel('r (Fraction of Overfitted Samples)')
    plt.ylabel('Metric Value')
    plt.title('AA Metrics vs r')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def generate_random_dnas(N, n):
    """Generate N random binary DNA sequences of length n.

    Args:
        N (int): Number of sequences.
        n (int): Sequence length.

    Returns:
        numpy.ndarray: Shape (N, n) with 0/1 entries.
    """
    return np.random.randint(0, 2, size=(N, n))

def generate_syn_dnas_from_train(x_train, m):
    """Generate synthetic DNA by copying first m bases from train samples.

    For each sequence, the first m positions are copied from the corresponding x_train
    row, and the remaining positions are random.

    Args:
        x_train (numpy.ndarray): Train DNA matrix shape (N, n).
        m (int): Number of prefix bases to copy.

    Returns:
        numpy.ndarray: Synthetic DNA shape (N, n).
    """
    N, n = x_train.shape
    x_syn = np.empty((N, n), dtype=int)
    # Copy first m bases from each train sample and randomize the remainder
    x_syn[:, :m] = x_train[:, :m]
    # Fill the rest with random 0s and 1s
    x_syn[:, m:] = np.random.randint(0, 2, size=(N, n - m))
    return x_syn

def dna_AA_simulation():
    N = 100
    n = 1000  # Length of each DNA sequence

    # Generate train, test, and synthetic DNA samples
    x_train = generate_random_dnas(N, n)
    x_test = generate_random_dnas(N, n)
    ms = np.arange(0.1, 1.0, 0.1)
    train_AA_list = []
    test_AA_list = []
    privacy_loss_list = []

    for frac in ms:
        m = int(frac * n)
        # x_synth = generate_random_dnas(N, n)
        x_synth = generate_syn_dnas_from_train(x_train, m)
        train_AA = calc_AA(x_train, x_synth, ord=1)
        test_AA = calc_AA(x_test, x_synth, ord=1)
        privacy_loss = test_AA - train_AA
        train_AA_list.append(train_AA)
        test_AA_list.append(test_AA)
        privacy_loss_list.append(privacy_loss)

    plt.figure(figsize=(8, 6))
    plt.plot(ms, train_AA_list, label='Train AA')
    plt.plot(ms, test_AA_list, label='Test AA')
    plt.plot(ms, privacy_loss_list, label='Privacy Loss')
    plt.xlabel('Fraction of Copied Bases (m/n)')
    plt.ylabel('Metric Value')
    plt.title('DNA AA Metrics vs Fraction of Copied Bases')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Call the function from main
if __name__ == "__main__":
    half_circle_AA_simulation()
    overfit_AA_graph()
    dna_AA_simulation()
