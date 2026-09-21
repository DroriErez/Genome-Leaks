"""Train an unconditional PCA diffusion model on 1000 Genomes haplotypes."""

import argparse
import datetime
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset

from dm_human_multi_configs import CONFIG_DM_HUMAN_MULTI, device_dm
from dm_model import DDPM, GaussianDiffusion, NoisePredictor, TimeSampler

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.device import get_gpu_memory_gb, get_memory_based_batch_size
from models.evaluate_model_quality import (
    allele_frequency_metrics,
    pca_embedding,
    pca_wasserstein_distance,
    plot_allele_frequencies,
    plot_pca,
    real_vs_synthetic_auc,
    save_metrics,
)


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


def read_hapt(path):
    """Read the two-metadata-column .hapt format into a binary matrix."""
    rows = []
    with Path(path).open("r") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            fields = line.split(maxsplit=2)
            if len(fields) != 3:
                raise ValueError(f"Missing genotype values at {path}:{line_number}")
            row = np.fromstring(fields[2], dtype=np.float32, sep=" ")
            if rows and row.size != rows[0].size:
                raise ValueError(f"Inconsistent SNP count at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"No haplotypes found in {path}")
    matrix = np.stack(rows)
    if not np.isin(matrix, (0, 1)).all():
        raise ValueError(f"{path} is not a binary haplotype matrix")
    return matrix


def reconstruct_binary(pca_scores, pca):
    """Map generated PCA scores back to hard 0/1 haplotype calls."""
    probabilities = np.clip(pca.inverse_transform(pca_scores), 0.0, 1.0)
    return (probabilities >= 0.5).astype(np.uint8)


def diffusion_loss(model, scores, batch_size, device):
    """Average DDPM noise-prediction loss without updating model weights."""
    loader = DataLoader(TensorDataset(torch.from_numpy(scores)), batch_size=batch_size)
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=True)
            total_loss += model.loss(batch).item() * len(batch)
    return total_loss / len(scores)


def find_last_checkpoint(checkpoint_dir):
    """Return the checkpoint with the highest epoch number, if one exists."""
    candidates = []
    for path in checkpoint_dir.glob("PCA_DM_model_*.pth"):
        match = re.fullmatch(r"PCA_DM_model_(\d+)\.pth", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    return max(candidates, default=(0, None), key=lambda item: item[0])


def inference_state_dict(model):
    """Create a CPU FP16 state dict for compact inference checkpoints."""
    return {
        name: tensor.detach().to(device="cpu", dtype=torch.float16)
        if tensor.is_floating_point() else tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
    }


def configure_for_available_memory():
    """Select a model size and batch size from the available GPU memory."""
    gpu_memory_gb = get_gpu_memory_gb()
    batch_size = get_memory_based_batch_size(
        small_batch_size=16,
        large_batch_size=256,
        large_gpu_memory_gb=16,
    )
    if gpu_memory_gb < 6:
        hidden_dims = (512, 1024, 1024)
    elif gpu_memory_gb < 12:
        hidden_dims = (1024, 2048, 2048)
    elif gpu_memory_gb < 16:
        hidden_dims = (2048, 4096, 4096)
    else:
        hidden_dims = (4096, 8192, 8192)

    (CONFIG_DM_HUMAN_MULTI["hidden_dim_1"],
     CONFIG_DM_HUMAN_MULTI["hidden_dim_2"],
     CONFIG_DM_HUMAN_MULTI["hidden_dim_3"]) = hidden_dims
    return gpu_memory_gb, batch_size, hidden_dims


def evaluate_checkpoint(
    real,
    synthetic,
    output_dir,
    epoch,
    training_loss=None,
    evaluation_loss=None,
):
    """Run the repository's genome-quality metrics for one checkpoint."""
    output_dir.mkdir(parents=True, exist_ok=True)
    n = min(len(real), len(synthetic))
    real = real[:n].astype(np.float32, copy=False)
    synthetic = synthetic[:n].astype(np.float32, copy=False)

    metrics = allele_frequency_metrics(real, synthetic)
    metrics.update({
        "model": "PCA_DM",
        "epoch": epoch,
        "training_loss": training_loss,
        "evaluation_loss": evaluation_loss,
        "n_real": n,
        "n_synthetic": n,
        "n_snps": real.shape[1],
    })

    try:
        from AA_Simulation.measurements import calc_AA
        aa_n = min(200, n)
        aa, real2real, real2synth, synth2synth = calc_AA(real[:aa_n], synthetic[:aa_n])
        metrics.update({
            "aa": float(aa),
            "real_to_real_distance_mean": float(np.mean(real2real)),
            "real_to_synth_distance_mean": float(np.mean(real2synth)),
            "synth_to_synth_distance_mean": float(np.mean(synth2synth)),
        })
    except ImportError as error:
        metrics["aa_status"] = f"skipped: {error}"

    metrics["real_vs_synthetic_classifier_auc"] = real_vs_synthetic_auc(
        real, synthetic, seed=SEED, pca_components=50
    )

    prefix = f"PCA_DM_{epoch}"
    allele_frequency_path = output_dir / f"{prefix}_allele_frequencies.csv"
    pd.DataFrame({
        "snp_index": np.arange(real.shape[1]),
        "real_frequency": metrics["real_af"],
        "synthetic_frequency": metrics["synth_af"],
    }).to_csv(allele_frequency_path, index=False)
    plot_allele_frequencies(
        metrics["real_af"], metrics["synth_af"],
        metrics["allele_frequency_mae"],
        output_dir / f"{prefix}_allele_frequency.png",
        title=f"Per-SNP allele frequency\nPCA-DM, epoch {epoch}",
    )

    combined = np.vstack((real, synthetic))
    labels = np.array(["Real"] * n + ["Synthetic"] * n)
    embedding = pca_embedding(combined, labels, SEED)
    embedding.insert(0, "model", "PCA_DM")
    embedding.insert(1, "epoch", epoch)
    metrics["pca_wasserstein_distance"] = pca_wasserstein_distance(embedding)
    embedding.to_csv(output_dir / f"{prefix}_pca2_embedding.csv", index=False)
    plot_pca(
        embedding,
        output_dir / f"{prefix}_pca2.png",
        wasserstein_dist=metrics["pca_wasserstein_distance"],
        title=f"Real vs synthetic genomes - PCA (2 components)\nPCA-DM, epoch {epoch}",
    )
    save_metrics(metrics, output_dir / f"{prefix}_quality_metrics.json")

    printable = {key: value for key, value in metrics.items()
                 if np.isscalar(value) and key != "epoch"}
    print(f"Checkpoint {epoch} evaluation results:")
    for key, value in printable.items():
        print(f"  {key}: {value}")
    return printable


def build_model(latent_dim, device):
    diffusion = GaussianDiffusion(CONFIG_DM_HUMAN_MULTI["num_timesteps"], device=device)
    time_sampler = TimeSampler(diffusion.tmin, diffusion.tmax)
    predictor = NoisePredictor(
        latent_dim,
        CONFIG_DM_HUMAN_MULTI["time_embedding_dim"],
        CONFIG_DM_HUMAN_MULTI["hidden_dim_1"],
        CONFIG_DM_HUMAN_MULTI["hidden_dim_2"],
        CONFIG_DM_HUMAN_MULTI["hidden_dim_3"],
        CONFIG_DM_HUMAN_MULTI["num_timesteps"],
    ).to(device)
    return DDPM(latent_dim, diffusion, time_sampler, predictor).to(device)


def parse_args():
    default_data = PROJECT_ROOT / "Data" / "1000G_real_genomes" / "10K_SNP_1000G_real_train.hapt"
    default_evaluation_data = PROJECT_ROOT / "Data" / "1000G_real_genomes" / "10K_SNP_1000G_real_test.hapt"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=default_data,
                        help="Binary 1000 Genomes .hapt training file")
    parser.add_argument("--evaluation-data-path", type=Path, default=default_evaluation_data,
                        help="Binary .hapt file used for checkpoint evaluation")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "checkpoints")
    parser.add_argument("--pca-components", type=int, default=CONFIG_DM_HUMAN_MULTI["snp_dim"])
    parser.add_argument("--epochs", type=int, default=CONFIG_DM_HUMAN_MULTI["num_epochs"])
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Training batch size; selected from GPU memory when omitted")
    parser.add_argument("--checkpoint-every", type=int, default=CONFIG_DM_HUMAN_MULTI["checkpoint"])
    parser.add_argument("--generate-samples", type=int, default=500)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                        help="Resume automatically from the highest saved epoch")
    return parser.parse_args()


def main():
    args = parse_args()
    gpu_memory_gb, automatic_batch_size, hidden_dims = configure_for_available_memory()
    if args.batch_size is None:
        args.batch_size = automatic_batch_size
    if device_dm.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(device_dm)} ({device_dm})")
        print(f"Detected GPU memory: {gpu_memory_gb:.2f} GB")
    else:
        print("Using CPU")
        print("Detected GPU memory: 0.00 GB")
    print(f"NoisePredictor hidden dimensions: {hidden_dims}")
    print(f"Training batch size: {args.batch_size}")
    haplotypes = read_hapt(args.data_path)
    evaluation_haplotypes = read_hapt(args.evaluation_data_path)
    if evaluation_haplotypes.shape[1] != haplotypes.shape[1]:
        raise ValueError("Training and evaluation files must contain the same number of SNPs")
    n_components = min(args.pca_components, haplotypes.shape[0], haplotypes.shape[1])
    if n_components < 1:
        raise ValueError("--pca-components must be positive")

    print(f"Loaded {haplotypes.shape[0]} binary haplotypes with {haplotypes.shape[1]} SNPs")
    print(f"Fitting PCA with {n_components} components")
    pca = PCA(n_components=n_components, random_state=SEED)
    scores = pca.fit_transform(haplotypes).astype(np.float32)
    evaluation_scores = pca.transform(evaluation_haplotypes).astype(np.float32)
    dataset = TensorDataset(torch.from_numpy(scores))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        pin_memory=device_dm.type == "cuda")

    model = build_model(n_components, device_dm)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG_DM_HUMAN_MULTI["optim_lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs * len(loader)),
        eta_min=CONFIG_DM_HUMAN_MULTI["optim_lr_min"],
    )

    run_dir = args.output_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "PCA_DM_training_losses.csv"
    if not log_path.exists():
        log_path.write_text("epoch,training_loss\n")
    results_path = run_dir / "PCA_DM_checkpoint_results.csv"
    if not results_path.exists():
        results_path.write_text(
            "epoch,training_loss,evaluation_loss,allele_frequency_mae,"
            "pca_wasserstein_distance\n"
        )

    saved_epoch, checkpoint_path = find_last_checkpoint(run_dir)
    if checkpoint_path is None:
        zero_prefix = "PCA_DM_0"
        np.savez(
            run_dir / f"{zero_prefix}_pca.npz",
            components=pca.components_,
            mean=pca.mean_,
            explained_variance=pca.explained_variance_,
            explained_variance_ratio=pca.explained_variance_ratio_,
        )
        np.save(
            run_dir / f"{zero_prefix}_training_allele_frequencies.npy",
            haplotypes.mean(axis=0),
        )
        zero_checkpoint = {
            "model_state_dict": inference_state_dict(model),
            "epoch": 0,
            "latent_dim": n_components,
            "snp_dim": haplotypes.shape[1],
            "config": CONFIG_DM_HUMAN_MULTI,
            "checkpoint_type": "inference",
            "weights_dtype": "float16",
        }
        zero_model_path = run_dir / "PCA_DM_model_0.pth"
        torch.save(zero_checkpoint, zero_model_path)
        print(f"Saved initial model checkpoint: {zero_model_path}")

    start_epoch = 1
    if args.resume and checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device_dm, weights_only=False)
        if checkpoint["latent_dim"] != n_components or checkpoint["snp_dim"] != haplotypes.shape[1]:
            raise ValueError(f"Checkpoint dimensions do not match current data: {checkpoint_path}")
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = saved_epoch + 1
        print(
            f"Loaded inference checkpoint {checkpoint_path}; resuming model weights "
            f"at epoch {start_epoch} with a new optimizer and scheduler"
        )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device_dm, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        average_loss = total_loss / len(loader)
        with log_path.open("a") as handle:
            handle.write(f"{epoch},{average_loss:.6f}\n")
        print(f"Epoch {epoch}/{args.epochs}: loss={average_loss:.6f}")

        if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
            prefix = f"PCA_DM_{epoch}"
            np.savez(
                run_dir / f"{prefix}_pca.npz",
                components=pca.components_,
                mean=pca.mean_,
                explained_variance=pca.explained_variance_,
                explained_variance_ratio=pca.explained_variance_ratio_,
            )
            np.save(
                run_dir / f"{prefix}_training_allele_frequencies.npy",
                haplotypes.mean(axis=0),
            )
            checkpoint = {
                "model_state_dict": inference_state_dict(model),
                "epoch": epoch,
                "latent_dim": n_components,
                "snp_dim": haplotypes.shape[1],
                "config": CONFIG_DM_HUMAN_MULTI,
                "checkpoint_type": "inference",
                "weights_dtype": "float16",
            }
            model_path = run_dir / f"PCA_DM_model_{epoch}.pth"
            torch.save(checkpoint, model_path)
            print(f"Saved model: {model_path}")
            model.eval()
            with torch.no_grad():
                generated_scores = model.sample(args.generate_samples, device_dm).cpu().numpy()
            synthetic = reconstruct_binary(generated_scores, pca)
            synthetic_path = (
                run_dir
                / f"PCA_DM_model_{epoch}_synthetic_{len(synthetic)}.npy"
            )
            np.save(synthetic_path, synthetic)
            print(f"Saved {len(synthetic)} cached synthetic haplotypes: {synthetic_path}")
            evaluation_loss = diffusion_loss(
                model, evaluation_scores, args.batch_size, device_dm
            )
            results = evaluate_checkpoint(
                evaluation_haplotypes,
                synthetic,
                run_dir,
                epoch,
                training_loss=average_loss,
                evaluation_loss=evaluation_loss,
            )
            with results_path.open("a") as handle:
                handle.write(
                    f"{epoch},{average_loss:.6f},{evaluation_loss:.6f},"
                    f"{results['allele_frequency_mae']:.6f},"
                    f"{results['pca_wasserstein_distance']:.6f}\n"
                )
            print(f"  evaluation_loss: {evaluation_loss}")
            print(f"Updated checkpoint summary: {results_path}")

    print(f"Training artifacts written to {run_dir}")


if __name__ == "__main__":
    main()
