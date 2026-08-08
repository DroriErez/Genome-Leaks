"""Membership inference using a diffusion model's noise-prediction loss."""

from typing import Any, Optional, Tuple
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from attack import attack
from utils.device import get_memory_based_batch_size


class DiffusionLossAttack(attack):
    name = "diffusion_loss_attack"

    def __init__(self, n_repeats=8, batch_size=None, timestep=None):
        super().__init__()
        self.timestep = None if timestep is None else int(timestep)
        if self.timestep is not None and self.timestep < 0:
            raise ValueError("timestep must be non-negative")
        if self.timestep is not None:
            self.name = f"diffusion_loss_attack_t{self.timestep}"
        self.n_repeats = max(1, int(n_repeats))
        self.batch_size = batch_size or get_memory_based_batch_size(
            small_batch_size=8,
            large_batch_size=64,
            large_gpu_memory_gb=16,
        )
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.score_threshold = None
        self.raw_score_threshold = None
        self.raw_score_threshold_0_5 = None
        self.non_train_losses = None
        self.non_member_scores = None
        self.last_raw_scores = None

    def get_display_name(self):
        if self.timestep is None:
            return "Denoising-Loss Attack"
        return f"Denoising-Loss Attack (t={self.timestep})"

    def is_attack_applicable(self, model):
        """Restrict diffusion-loss scoring to PCA-DM model wrappers."""
        return model.get_model_architecture() == "PCA_DM"

    def fit(self, non_train_data, thr=0.99, modelWrapper=None):
        if modelWrapper is None or not hasattr(modelWrapper, "diffusion_losses"):
            raise ValueError("A PCA-DM wrapper with diffusion_losses is required")
        self.modelWrapper = modelWrapper
        self.threshold = thr
        self.non_train_data = self._as_2d(non_train_data)
        if len(self.non_train_data) < 1:
            raise ValueError("non_train_data must contain at least one sample")
        self.non_train_losses = self._compute_losses(self.non_train_data)
        self.non_member_scores = self._losses_to_scores(self.non_train_losses)
        percentile = self._threshold_to_percentile(thr)
        self.score_threshold = np.percentile(self.non_member_scores, percentile)
        # Diffusion loss has the opposite direction from membership score:
        # smaller raw losses are more member-like.
        self.raw_score_threshold = np.percentile(
            self.non_train_losses, 100 - percentile
        )
        self.raw_score_threshold_0_5 = np.percentile(self.non_train_losses, 50)
        print(
            "Fitted denoising-loss attack "
            f"with raw non-member mean={np.mean(self.non_train_losses):.6f}, "
            f"std={np.std(self.non_train_losses):.6f}, "
            f"score threshold={self.score_threshold:.4f}, "
            f"raw loss threshold={self.raw_score_threshold:.6f}, "
            f"raw loss threshold @ 0.5={self.raw_score_threshold_0_5:.6f}"
        )
        self.is_fitted = True
        return

    @staticmethod
    def _as_2d(data):
        data = np.asarray(data)
        if data.ndim == 1:
            return data.reshape(1, -1)
        return data.reshape(data.shape[0], -1)

    @staticmethod
    def _threshold_to_percentile(threshold):
        if threshold is None:
            raise ValueError("threshold percentile must be provided")
        if 0 <= threshold <= 1:
            return threshold * 100
        if 0 <= threshold <= 100:
            return threshold
        raise ValueError("threshold percentile must be between 0 and 100")

    def _compute_losses(self, data):
        return self.modelWrapper.diffusion_losses(
            data,
            n_repeats=self.n_repeats,
            batch_size=self.batch_size,
            timestep=self.timestep,
        )

    def _losses_to_scores(self, losses):
        return np.asarray([
            np.mean(self.non_train_losses >= loss) for loss in losses
        ])

    def score(self, candidates: np.ndarray, **kwargs: Any) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Attack must be fitted before scoring")
        self.last_raw_scores = self._compute_losses(self._as_2d(candidates))
        return self._losses_to_scores(self.last_raw_scores)

    def raw_score(self, candidates: np.ndarray) -> np.ndarray:
        """Return per-sample diffusion losses before percentile calibration."""
        if not self.is_fitted:
            raise ValueError("Attack must be fitted before scoring")
        return self._compute_losses(self._as_2d(candidates))

    def diffusion_loss_diagnostic(
        self,
        train_data: np.ndarray,
        non_train_data: np.ndarray,
        synthetic_data: np.ndarray,
        model_name: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> dict:
        """Summarize and save raw diffusion losses for the main data splits."""
        split_losses = {
            "train": self.raw_score(train_data),
            "non_train": self.raw_score(non_train_data),
            "synthetic": self.raw_score(synthetic_data),
        }
        summary = {
            "model": model_name,
            "n_repeats": self.n_repeats,
            "timestep": self.timestep,
        }
        for split, losses in split_losses.items():
            summary[f"n_{split}"] = len(losses)
            for statistic, function in (
                ("mean", np.mean),
                ("median", np.median),
                ("std", np.std),
                ("min", np.min),
                ("max", np.max),
            ):
                summary[f"{split}_{statistic}"] = function(losses)

        if output_dir is not None:
            if model_name is None:
                raise ValueError("model_name must be provided when output_dir is set")
            output_dir = Path(output_dir)
            csv_path = output_dir / f"{model_name}_{self.name}_diagnostic.csv"
            rows = []
            for split, losses in split_losses.items():
                rows.extend(
                    {
                        "model": model_name,
                        "split": split,
                        "sample_index": index,
                        "average_diffusion_loss": loss,
                        "n_repeats": self.n_repeats,
                        "timestep": self.timestep,
                    }
                    for index, loss in enumerate(losses)
                )
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            summary["path"] = csv_path

            plot_path = output_dir / f"{model_name}_{self.name}_diagnostic_distribution.png"
            self._plot_loss_distributions(split_losses, plot_path)
            summary["plot_path"] = plot_path
        return summary

    def _plot_loss_distributions(self, split_losses, output_path):
        all_losses = np.concatenate(list(split_losses.values()))
        bins = 1 if len(np.unique(all_losses)) < 2 else np.linspace(
            np.min(all_losses), np.max(all_losses), 101
        )
        plt.figure(figsize=(8, 5))
        for label, key, color in (
            ("Train", "train", "tab:blue"),
            ("Non-train", "non_train", "tab:orange"),
            ("Synthetic", "synthetic", "tab:green"),
        ):
            plt.hist(
                split_losses[key], bins=bins, density=True, histtype="step",
                linewidth=2, label=label, color=color,
            )
        plt.xlabel("Average Denoising-Loss")
        plt.ylabel("Density")
        plt.title(f"{self.get_display_name()} Distributions")
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

    def predict(
        self,
        candidates: np.ndarray,
        threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        scores = self.score(candidates, **kwargs)
        decision_threshold = self.score_threshold if threshold is None else threshold
        if decision_threshold is None:
            raise ValueError("No threshold available. Call fit(...) or pass threshold.")
        return scores >= decision_threshold, scores
