"""Evaluate generated genome quality with allele-frequency and PCA plots.

Example:
    python models/evaluate_model_quality.py ^
        --model attacks/models_to_attack/VAE_model_10000.pth ^
        --real attacks/models_to_attack/VAE_model_10000_test.hapt ^
        --output-dir attacks/results/model_quality ^
        --n-samples 1000
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import wasserstein_distance

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from AA_Simulation.measurements import calc_AA
from models.models_factory import create_model_wrapper
from utils.device import get_device, get_memory_based_batch_size


PLOT_LABEL_FONT_SIZE = 22
PLOT_TITLE_FONT_SIZE = 20
PLOT_LEGEND_FONT_SIZE = 20
PLOT_TICK_FONT_SIZE = 18
PLOT_CALLOUT_FONT_SIZE = 20
PLOT_FIG_SIZE = (6.5, 5.7)


def read_hapt(path, nrows=None):
    """Read a .hapt-like file and return only SNP columns."""
    df = pd.read_csv(path, sep=" ", header=None, nrows=nrows)
    if df.shape[1] <= 2:
        raise ValueError(f"{path} must contain at least two metadata columns and SNP columns")
    return df.drop(df.columns[0:2], axis=1).to_numpy(dtype=np.float32)


def sample_rows(data, n, seed):
    if n is None or n >= len(data):
        return data
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(data), size=n, replace=False)
    return data[idx]


def allele_frequency_metrics(real, synth):
    real_af = real.mean(axis=0)
    synth_af = synth.mean(axis=0)
    diff = synth_af - real_af

    pearson = np.corrcoef(real_af, synth_af)[0, 1]
    spearman = pd.Series(real_af).corr(pd.Series(synth_af), method="spearman")

    return {
        "real_af": real_af,
        "synth_af": synth_af,
        "allele_frequency_mae": float(np.mean(np.abs(diff))),
        "allele_frequency_mse": float(np.mean(diff**2)),
        "allele_frequency_rmse": float(np.sqrt(np.mean(diff**2))),
        "allele_frequency_max_abs_error": float(np.max(np.abs(diff))),
        "allele_frequency_pearson": float(np.nan_to_num(pearson)),
        "allele_frequency_spearman": float(np.nan_to_num(spearman)),
        "real_polymorphic_snp_fraction": float(np.mean((real_af > 0) & (real_af < 1))),
        "synth_polymorphic_snp_fraction": float(np.mean((synth_af > 0) & (synth_af < 1))),
    }


def plot_allele_frequencies(
    real_af,
    synth_af,
    mae,
    output_path,
    title="Per-SNP allele frequency",
):
    fig, ax = plt.subplots(figsize=PLOT_FIG_SIZE)

    ax.scatter(real_af, synth_af, s=8, alpha=0.25, label="SNPs")
    ax.plot([0, 1], [0, 1], color="black", linewidth=1, linestyle="--", label="Perfect match")
    ax.set_xlabel("Real allele frequency", fontsize=PLOT_LABEL_FONT_SIZE)
    ax.set_ylabel("Synthetic allele frequency", fontsize=PLOT_LABEL_FONT_SIZE)
    ax.set_title(title, fontsize=PLOT_TITLE_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=PLOT_TICK_FONT_SIZE)
    ax.legend(loc="upper left", fontsize=PLOT_LEGEND_FONT_SIZE)
    ax.grid(alpha=0.2)
    ax.text(
        0.98,
        0.02,
        f"MAE = {mae:.4f}",
        transform=ax.transAxes,
        horizontalalignment="right",
        verticalalignment="bottom",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "black", "alpha": 0.85},
        fontsize=PLOT_CALLOUT_FONT_SIZE,
    )

    # Frequency-distribution histogram intentionally disabled.
    # ax.hist(real_af, bins=50, alpha=0.6, label="Real", density=True)
    # ax.hist(synth_af, bins=50, alpha=0.6, label="Synthetic", density=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def umap_embedding(data, labels, seed, metric="hamming", pca_components=None):
    try:
        from umap import UMAP
    except ImportError as exc:
        raise ImportError(
            "UMAP plotting requires umap-learn. Install it with: pip install umap-learn"
        ) from exc

    umap_input = data
    if pca_components is not None:
        max_components = min(pca_components, data.shape[0] - 1, data.shape[1])
        umap_input = make_pipeline(
            StandardScaler(),
            PCA(n_components=max_components, random_state=seed),
        ).fit_transform(data)

    reducer = UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.1,
        metric=metric,
        random_state=seed,
    )
    embedding = reducer.fit_transform(umap_input)
    return pd.DataFrame(
        {
            "UMAP1": embedding[:, 0],
            "UMAP2": embedding[:, 1],
            "source": labels,
        }
    )


def plot_umap(embedding_df, output_path, title="Real vs synthetic genomes"):
    fig, ax = plt.subplots(figsize=PLOT_FIG_SIZE)
    for source, color in [("Real", "tab:red"), ("Synthetic", "tab:blue")]:
        rows = embedding_df["source"] == source
        ax.scatter(
            embedding_df.loc[rows, "UMAP1"],
            embedding_df.loc[rows, "UMAP2"],
            s=16,
            alpha=0.35,
            label=source,
            color=color,
        )
    ax.set_xlabel("UMAP1", fontsize=PLOT_LABEL_FONT_SIZE)
    ax.set_ylabel("UMAP2", fontsize=PLOT_LABEL_FONT_SIZE)
    ax.set_title(title, fontsize=PLOT_TITLE_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=PLOT_TICK_FONT_SIZE)
    ax.legend(fontsize=PLOT_LEGEND_FONT_SIZE)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def pca_embedding(data, labels, seed):
    """Project real and synthetic genomes to two PCA components for visualization."""
    embedding = make_pipeline(
        StandardScaler(),
        PCA(n_components=2, random_state=seed),
    ).fit_transform(data)
    return pd.DataFrame(
        {
            "PCA1": embedding[:, 0],
            "PCA2": embedding[:, 1],
            "source": labels,
        }
    )


def pca_wasserstein_distance(embedding_df):
    """Return the combined 1D Wasserstein distance across the two PCA axes."""
    real = embedding_df[embedding_df["source"] == "Real"]
    synth = embedding_df[embedding_df["source"] == "Synthetic"]
    component_distances = [
        wasserstein_distance(real[component], synth[component])
        for component in ("PCA1", "PCA2")
    ]
    return float(np.linalg.norm(component_distances))


def plot_pca(
    embedding_df,
    output_path,
    wasserstein_dist,
    title="Real vs synthetic genomes - PCA (2 components)",
):
    fig, ax = plt.subplots(figsize=PLOT_FIG_SIZE)
    for source, color in [("Real", "tab:red"), ("Synthetic", "tab:blue")]:
        rows = embedding_df["source"] == source
        ax.scatter(
            embedding_df.loc[rows, "PCA1"],
            embedding_df.loc[rows, "PCA2"],
            s=16,
            alpha=0.35,
            label=source,
            color=color,
        )
    ax.set_xlabel("PCA1", fontsize=PLOT_LABEL_FONT_SIZE)
    ax.set_ylabel("PCA2", fontsize=PLOT_LABEL_FONT_SIZE)
    ax.set_title(title, fontsize=PLOT_TITLE_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=PLOT_TICK_FONT_SIZE)
    ax.legend(fontsize=PLOT_LEGEND_FONT_SIZE)
    ax.grid(alpha=0.2)
    ax.text(
        0.98,
        0.02,
        f"W distance = {wasserstein_dist:.3f}",
        transform=ax.transAxes,
        horizontalalignment="right",
        verticalalignment="bottom",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "black", "alpha": 0.85},
        fontsize=PLOT_CALLOUT_FONT_SIZE,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def real_vs_synthetic_auc(real, synth, seed, pca_components):
    n = min(len(real), len(synth))
    x = np.vstack([real[:n], synth[:n]])
    y = np.concatenate([np.ones(n), np.zeros(n)])

    stratify = y if n >= 5 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.3,
        random_state=seed,
        stratify=stratify,
    )

    max_components = min(pca_components, x_train.shape[0] - 1, x_train.shape[1])
    model = make_pipeline(
        StandardScaler(),
        PCA(n_components=max_components, random_state=seed),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    model.fit(x_train, y_train)
    scores = model.predict_proba(x_test)[:, 1]
    auc = roc_auc_score(y_test, scores)
    return float(max(auc, 1 - auc))


def save_metrics(metrics, output_path):
    serializable = {
        key: value
        for key, value in metrics.items()
        if not isinstance(value, np.ndarray)
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    pd.DataFrame([serializable]).to_csv(output_path.with_suffix(".csv"), index=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Save allele-frequency and PCA quality plots for a genome generator."
    )
    parser.add_argument("--model", required=True, help="Path to a saved model checkpoint")
    parser.add_argument("--real", required=True, help="Path to the real dataset to compare against")
    parser.add_argument("--output-dir", default="attacks/results/model_quality")
    parser.add_argument("--n-samples", type=int, default=1000, help="Rows to read/generate")
    parser.add_argument("--pca-plot-samples", type=int, default=1000, help="Rows per source for PCA plot")
    parser.add_argument("--aa-samples", type=int, default=200, help="Rows per source for AA metric")
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--generation-batch-size", type=int, default=None)
    parser.add_argument("--device", default="auto", help="Torch device: auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def evaluate_model_quality(
    model,
    real_path,
    output_dir="attacks/results/model_quality",
    n_samples=1000,
    pca_plot_samples=1000,
    aa_samples=200,
    pca_components=50,
    seed=42,
):
    """Generate quality plots and metrics for an already-loaded model wrapper."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(model, "load_attack_dataset"):
        real = model.load_attack_dataset(real_path, nrows=n_samples)
    else:
        real = read_hapt(real_path, nrows=n_samples)
    real = np.asarray(real, dtype=np.float32)
    synth = model.generate(n=len(real)).astype(np.float32, copy=False)
    prefix = model.model_name

    if synth.shape[1] != real.shape[1]:
        min_width = min(synth.shape[1], real.shape[1])
        print(
            f"Warning: real has {real.shape[1]} SNPs and synthetic has {synth.shape[1]}; "
            f"using first {min_width} columns."
        )
        real = real[:, :min_width]
        synth = synth[:, :min_width]

    metrics = allele_frequency_metrics(real, synth)
    metrics.update(
        {
            "model": model.model_name,
            "real_data": str(real_path),
            "n_real": int(len(real)),
            "n_synthetic": int(len(synth)),
            "n_snps": int(real.shape[1]),
        }
    )

    aa_n = min(aa_samples, len(real), len(synth))
    if aa_n >= 2:
        aa, real2real, real2synth, synth2synth = calc_AA(real[:aa_n], synth[:aa_n])
        metrics.update(
            {
                "aa": float(aa),
                "real_to_real_distance_mean": float(np.mean(real2real)),
                "real_to_synth_distance_mean": float(np.mean(real2synth)),
                "synth_to_synth_distance_mean": float(np.mean(synth2synth)),
            }
        )

    metrics["real_vs_synthetic_classifier_auc"] = real_vs_synthetic_auc(
        real,
        synth,
        seed=seed,
        pca_components=pca_components,
    )

    title_suffix = model.get_model_title_suffix()
    allele_plot_path = output_dir / f"{prefix}_allele_frequency.png"
    plot_allele_frequencies(
        metrics["real_af"],
        metrics["synth_af"],
        metrics["allele_frequency_mae"],
        allele_plot_path,
        title=f"Per-SNP allele frequency\n{title_suffix}",
    )

    # UMAP calculations and plots are disabled in favor of a direct 2D PCA projection.
    pca_n = min(pca_plot_samples, len(real), len(synth))
    real_pca = sample_rows(real, pca_n, seed)
    synth_pca = sample_rows(synth, pca_n, seed + 1)
    combined = np.vstack([real_pca, synth_pca]).astype(np.float32, copy=False)
    labels = np.array(["Real"] * len(real_pca) + ["Synthetic"] * len(synth_pca))
    pca_embedding_df = pca_embedding(combined, labels, seed)
    pca_w_distance = pca_wasserstein_distance(pca_embedding_df)
    metrics["pca_wasserstein_distance"] = pca_w_distance
    pca_plot_path = output_dir / f"{prefix}_pca2.png"
    pca_csv_path = output_dir / f"{prefix}_pca2_embedding.csv"
    plot_pca(
        pca_embedding_df,
        pca_plot_path,
        wasserstein_dist=pca_w_distance,
        title=f"Real vs synthetic genomes - PCA\n{title_suffix}",
    )
    pca_embedding_df.to_csv(pca_csv_path, index=False)

    metrics_path = output_dir / f"{prefix}_quality_metrics.json"
    save_metrics(metrics, metrics_path)

    print(f"Saved allele-frequency plot: {allele_plot_path}")
    print(f"Saved PCA (2 components) plot: {pca_plot_path}")
    print(f"Saved PCA (2 components) embedding: {pca_csv_path}")
    print(f"Saved quality metrics: {metrics_path}")
    print(f"Saved quality metrics CSV: {metrics_path.with_suffix('.csv')}")
    return {
        "allele_frequency_plot": allele_plot_path,
        "pca2_plot": pca_plot_path,
        "pca2_embedding": pca_csv_path,
        "metrics_json": metrics_path,
        "metrics_csv": metrics_path.with_suffix(".csv"),
        "metrics": {
            key: value
            for key, value in metrics.items()
            if not isinstance(value, np.ndarray)
        },
    }


def main():
    args = parse_args()
    device = get_device() if args.device == "auto" else torch.device(args.device)
    generation_batch_size = args.generation_batch_size
    if generation_batch_size is None:
        generation_batch_size = get_memory_based_batch_size(
            small_batch_size=8,
            large_batch_size=512,
            large_gpu_memory_gb=16,
        )
    model = create_model_wrapper(
        args.model,
        device=device,
        generation_batch_size=generation_batch_size,
    )
    evaluate_model_quality(
        model=model,
        real_path=args.real,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        pca_plot_samples=args.pca_plot_samples,
        aa_samples=args.aa_samples,
        pca_components=args.pca_components,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
