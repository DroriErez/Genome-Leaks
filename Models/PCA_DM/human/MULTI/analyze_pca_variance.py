"""Report retained variance from PCA-DM checkpoint PCA files."""

import argparse
from pathlib import Path

import numpy as np


def analyze_pca_file(path, thresholds=(0.80, 0.90, 0.95, 0.99)):
    with np.load(path) as pca_data:
        if "explained_variance_ratio" not in pca_data:
            raise KeyError(f"{path} does not contain explained_variance_ratio")
        ratios = np.asarray(pca_data["explained_variance_ratio"], dtype=np.float64)

    cumulative = np.cumsum(ratios)
    print(f"\n{path.name}")
    print(f"  saved components: {len(ratios)}")
    print(f"  retained variance: {cumulative[-1]:.4%}")
    for threshold in thresholds:
        index = int(np.searchsorted(cumulative, threshold, side="left"))
        if index < len(cumulative):
            print(f"  components for {threshold:.0%}: {index + 1}")
        else:
            print(f"  components for {threshold:.0%}: more than {len(ratios)}")
    return cumulative[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint_dir",
        nargs="?",
        type=Path,
        default=Path(__file__).parent / "checkpoints",
        help="Directory containing PCA_DM_<epoch>_pca.npz files",
    )
    args = parser.parse_args()
    paths = sorted(args.checkpoint_dir.glob("PCA_DM_*_pca.npz"))
    if not paths:
        raise FileNotFoundError(f"No PCA_DM_*_pca.npz files found in {args.checkpoint_dir}")

    retained = [analyze_pca_file(path) for path in paths]
    if len(paths) > 1 and np.allclose(retained, retained[0]):
        print("\nAll checkpoint PCA files retain the same variance, as expected.")


if __name__ == "__main__":
    main()
