"""
Simple reconstruction-loss membership inference attack.

This attack is intended for autoencoder-style generative models, currently
restricted to VAE wrappers by ``is_attack_applicable``. It reconstructs each
candidate with the target model and uses the per-sample reconstruction loss as
the privacy signal: samples that reconstruct unusually well are treated as more
likely training members.

The public score follows the convention used by the attack evaluation pipeline:
higher scores mean "more likely to be a member." Internally, the class computes
binary cross-entropy reconstruction losses and converts each loss into the
fraction of non-training reference losses that are greater than or equal to it.
"""

import numpy as np
import pandas as pd
import torch
from typing import Any, Optional, Tuple
from pathlib import Path
from torch.nn import functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from attack import attack
from utils.device import get_memory_based_batch_size

class ReconstructionLossAttack(attack):
    """
    Membership inference attack based on model reconstruction loss.

    The attack assumes that a generative model may reconstruct training samples
    better than non-training samples. During fitting, it stores reconstruction
    losses for non-training samples. During scoring, it
    computes the candidate reconstruction loss and compares it with the stored
    non-training loss distribution.

    Attributes:
        name: Attack identifier used by the attack runner and result files.
        threshold: Decision threshold on membership scores. Defaults to the
            score percentile calibrated from non-training samples during
            ``fit``.
        modelWrapper: Wrapper that owns the target model. The wrapper must expose
            a ``model`` attribute that accepts tensors shaped as
            ``(batch_size, 1, sequence_length)`` and returns reconstructions in
            the same shape.
        non_train_losses: Reconstruction losses computed for non-training
            reference samples during fitting.
    """
    name = "reconstruction_loss_attack"

    def get_display_name(self) -> str:
        return "Reconstruction Loss Attack"

    def _as_2d(self, data):
        data = np.asarray(data)
        if data.ndim == 1:
            return data.reshape(1, -1)
        return data.reshape(data.shape[0], -1)

    def fit(
        self,
        non_train_data: np.ndarray,
        thr: float = 0.5,
        modelWrapper=None,
        n_repeats: int = 3,
    ) -> None:
        """
        Prepare the attack by computing reference reconstruction losses.

        This attack does not train a separate attacker model. It stores the
        provided threshold, keeps the target model wrapper, and delegates to
        ``_fit`` to evaluate reference losses.

        Args:
            non_train_data: Known non-member samples with shape
                ``(n_samples, sequence_length)``. These losses are stored for
                inspection and possible threshold analysis.
            thr: Non-training score percentile used for threshold calibration.
                For example, ``thr=0.99`` sets the score threshold at the 99th
                percentile and implies an FPR target of ``1 - thr``.
            modelWrapper: Target generative model wrapper. Required for scoring.
            n_repeats: Number of stochastic reconstruction attempts to average
                for each sample.

        Raises:
            ValueError: If no target model wrapper is provided.
        """
        if modelWrapper is None or getattr(modelWrapper, "model", None) is None:
            raise ValueError("modelWrapper with a loaded model is required")

        self.threshold = thr
        self.modelWrapper = modelWrapper
        self.n_repeats = max(1, int(n_repeats))
        self._fit(
            non_train_data=non_train_data,
            modelWrapper=modelWrapper,
        )
        self.is_fitted = True
        return

    def _fit(
        self,
        non_train_data: np.ndarray,
        modelWrapper=None,
    ) -> None:
        """
        Compute and store reconstruction losses for the non-training reference dataset.

        The model is switched to evaluation mode before losses are computed.
        Batching keeps memory usage bounded when fitting on larger arrays.

        Args:
            non_train_data: Known non-member reference samples.
            modelWrapper: Target model wrapper passed through from ``fit``.
        """
        model = modelWrapper.model if modelWrapper is not None else self.modelWrapper.model
        model.eval()

        self.batch_eval_size = get_memory_based_batch_size(
            small_batch_size=16,
            large_batch_size=256,
            large_gpu_memory_gb=16,
        )
        print(f"Using reconstruction loss batch size: {self.batch_eval_size}")

        self.non_train_losses = self._compute_reconstruction_losses(
            non_train_data,
            batch_size=self.batch_eval_size,
        )

        self.non_member_scores = self._losses_to_scores(self.non_train_losses)
        self.fpr_target = self._threshold_to_fpr(self.threshold)
        self.score_threshold = np.percentile(
            self.non_member_scores,
            self._threshold_to_percentile(self.threshold),
        )
        print(
            "Fitted reconstruction loss attack "
            f"with FPR target: {self.fpr_target:.4f}, "
            f"score threshold: {self.score_threshold:.4f}"
        )

        return None

    def _threshold_to_percentile(self, threshold):
        if threshold is None:
            raise ValueError("threshold percentile must be provided")
        if 0 <= threshold <= 1:
            return threshold * 100
        if 0 <= threshold <= 100:
            return threshold
        raise ValueError("threshold percentile must be between 0 and 1 or 0 and 100")

    def _threshold_to_fpr(self, threshold):
        percentile = self._threshold_to_percentile(threshold)
        return 1 - (percentile / 100)

    def _losses_to_scores(self, losses):
        return np.array([
            np.mean(self.non_train_losses >= loss)
            for loss in losses
        ])

    def _compute_reconstruction_losses(
        self,
        data: np.ndarray,
        batch_size: int = 128,
    ) -> np.ndarray:
        """
        Compute binary cross-entropy reconstruction loss for each sample.
        For stochastic models, each sample loss is averaged over
        ``self.n_repeats`` reconstruction attempts.

        Args:
            data: Candidate or reference samples shaped as
                ``(n_samples, sequence_length)``.
            batch_size: Number of samples to reconstruct per forward pass.

        Returns:
            A one-dimensional NumPy array where each value is the summed BCE
            reconstruction loss for the matching input sample.

        Raises:
            ValueError: If ``data`` is empty or the model wrapper is unavailable.
        """
        if self.modelWrapper is None or getattr(self.modelWrapper, "model", None) is None:
            raise ValueError("modelWrapper with a loaded model is required")

        if len(data) == 0:
            raise ValueError("data must contain at least one sample")

        model = self.modelWrapper.model
        model_device = next(model.parameters()).device

        x = torch.as_tensor(data, device=model_device).float()
        x = x.reshape(x.shape[0], 1, x.shape[1])

        losses = []
        n_repeats = getattr(self, "n_repeats", 3)

        with torch.no_grad():
            for i in range(0, x.shape[0], batch_size):
                batch = x[i:i + batch_size]
                repeated_losses = []

                for _ in range(n_repeats):
                    x_hat = model(batch)

                    # BCE per element
                    per_element_loss = F.binary_cross_entropy(
                        x_hat,
                        batch,
                        reduction="none",
                    )

                    # loss per sample: sum over all non-batch dimensions
                    per_sample_loss = per_element_loss.view(batch.shape[0], -1).sum(dim=1)
                    repeated_losses.append(per_sample_loss)

                avg_per_sample_loss = torch.stack(repeated_losses).mean(dim=0)
                losses.append(avg_per_sample_loss.detach().cpu().numpy())

        return np.concatenate(losses)

    def is_attack_applicable(self, model) -> bool:
        """
        Return whether the attack can be applied to the given model wrapper.

        The current implementation expects a VAE-like wrapper because it relies
        on direct reconstruction through ``modelWrapper.model(batch)``.
        """
        return model.get_model_architecture() == "VAE"

    def score(self, candidates: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Compute membership scores for candidate samples.

        A low reconstruction loss is considered more member-like. To align with
        the shared attack API, this method converts losses into scores where
        higher values indicate stronger membership evidence:

        ``score = mean(non_train_losses >= candidate_loss)``

        Args:
            candidates: Input samples to score with shape
                ``(n_samples, sequence_length)``.
            **kwargs: Additional arguments accepted for API compatibility.

        Returns:
            One score per candidate. Scores are in ``[0, 1]``; higher scores
            mean the candidate had lower reconstruction loss relative to the
            non-training reference distribution.

        Raises:
            ValueError: If no model wrapper was provided during fitting.
        """
        if self.modelWrapper is None:
            raise ValueError("Model must be provided for reconstruction loss attack")

        if not hasattr(self, "non_train_losses"):
            raise ValueError("Attack must be fitted before scoring")

        query_losses = self._compute_reconstruction_losses(
            candidates,
            batch_size=getattr(self, "batch_eval_size", 128),
        )
        return self._losses_to_scores(query_losses)

    def reconstruction_loss_diagnostic(
        self,
        train_data: np.ndarray,
        non_train_data: np.ndarray,
        synthetic_data: np.ndarray,
        model_name: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> dict:
        """Summarize and save per-sample reconstruction losses for key splits."""
        batch_size = getattr(self, "batch_eval_size", 128)
        train_losses = self._compute_reconstruction_losses(
            train_data,
            batch_size=batch_size,
        )
        non_train_losses = self._compute_reconstruction_losses(
            non_train_data,
            batch_size=batch_size,
        )
        synthetic_losses = self._compute_reconstruction_losses(
            synthetic_data,
            batch_size=batch_size,
        )

        summary = {
            "model": model_name,
            "n_train": len(train_losses),
            "n_non_train": len(non_train_losses),
            "n_synthetic": len(synthetic_losses),
            "train_mean": np.mean(train_losses),
            "train_median": np.median(train_losses),
            "train_std": np.std(train_losses),
            "train_min": np.min(train_losses),
            "train_max": np.max(train_losses),
            "non_train_mean": np.mean(non_train_losses),
            "non_train_median": np.median(non_train_losses),
            "non_train_std": np.std(non_train_losses),
            "non_train_min": np.min(non_train_losses),
            "non_train_max": np.max(non_train_losses),
            "synthetic_mean": np.mean(synthetic_losses),
            "synthetic_median": np.median(synthetic_losses),
            "synthetic_std": np.std(synthetic_losses),
            "synthetic_min": np.min(synthetic_losses),
            "synthetic_max": np.max(synthetic_losses),
        }

        if output_dir is not None:
            if model_name is None:
                raise ValueError("model_name must be provided when output_dir is set")

            out_path = Path(output_dir) / f"{model_name}_{self.name}_diagnostic.csv"
            diagnostic_rows = []
            for split_name, split_losses in (
                ("train", train_losses),
                ("non_train", non_train_losses),
                ("synthetic", synthetic_losses),
            ):
                for sample_index, loss in enumerate(split_losses):
                    diagnostic_rows.append({
                        "model": model_name,
                        "split": split_name,
                        "sample_index": sample_index,
                        "average_reconstruction_loss": loss,
                    })
            pd.DataFrame(diagnostic_rows).to_csv(out_path, index=False)
            summary["path"] = out_path

            plot_path = Path(output_dir) / f"{model_name}_{self.name}_diagnostic_distribution.png"
            self._plot_reconstruction_loss_distributions(
                train_losses,
                non_train_losses,
                synthetic_losses,
                plot_path,
            )
            summary["plot_path"] = plot_path

        return summary

    def _plot_reconstruction_loss_distributions(
        self,
        train_losses: np.ndarray,
        non_train_losses: np.ndarray,
        synthetic_losses: np.ndarray,
        output_path: Path,
    ) -> None:
        """Save overlaid density lines for reconstruction-loss distributions."""
        all_losses = np.concatenate([
            train_losses,
            non_train_losses,
            synthetic_losses,
        ])
        if len(np.unique(all_losses)) < 2:
            bins = 1
        else:
            bins = np.linspace(np.min(all_losses), np.max(all_losses), 101)

        plt.figure(figsize=(8, 5))
        for label, losses, color in (
            ("Train", train_losses, "tab:blue"),
            ("Non-train", non_train_losses, "tab:orange"),
            ("Synthetic", synthetic_losses, "tab:green"),
        ):
            density, edges = np.histogram(losses, bins=bins, density=True)
            centers = (edges[:-1] + edges[1:]) / 2
            smoothed_density = self._smooth_density(density)
            plt.plot(
                centers,
                density,
                color=color,
                linestyle=":",
                linewidth=1.2,
                alpha=0.65,
            )
            plt.plot(
                centers,
                smoothed_density,
                label=f"{label} smoothed",
                color=color,
                linewidth=2.2,
            )

        plt.xlabel("Average Reconstruction Loss")
        plt.ylabel("Density")
        plt.title(f"{self.get_display_name()} Distributions")
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

    @staticmethod
    def _smooth_density(density: np.ndarray) -> np.ndarray:
        if len(density) < 5:
            return density

        kernel = np.array([1, 4, 7, 10, 7, 4, 1], dtype=np.float64)
        kernel = kernel / np.sum(kernel)
        padded_density = np.pad(density, (len(kernel) // 2,), mode="edge")
        return np.convolve(padded_density, kernel, mode="valid")

    def predict(
        self,
        candidates: np.ndarray,
        threshold: Optional[float] = None,
        model=None,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict membership labels for candidate samples.

        Args:
            candidates: Input samples with shape
                ``(n_samples, sequence_length)``.
            threshold: Optional decision threshold on scores. If omitted, the
                score threshold calibrated during ``fit`` is used.
            model: Optional model argument kept for compatibility with other
                attacks. The fitted ``modelWrapper`` is used for reconstruction.
            **kwargs: Additional arguments forwarded to ``score``.

        Returns:
            A tuple ``(predictions, scores)``. ``predictions`` is a boolean array
            where ``True`` means predicted member, and ``scores`` contains the
            raw membership scores.

        Raises:
            ValueError: If no threshold was supplied and the attack has not been
                fitted with a threshold.
        """
        scores = self.score(candidates, **kwargs)
        thr = self.score_threshold if threshold is None else threshold

        if thr is None:
            raise ValueError("No threshold available. Call fit(...) or pass threshold.")

        predictions = (scores >= thr)

        return predictions, scores
