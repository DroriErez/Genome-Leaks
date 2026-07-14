"""Critic-score membership inference attack for WGAN models.

The attack uses the WGAN critic as the privacy signal. Higher critic values are
treated as stronger evidence that a sample resembles the training distribution.
To make scores comparable, raw critic outputs are converted into percentile-like
membership scores against known non-training samples.
"""

from typing import Any, Optional, Tuple
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from attack import attack


class CriticScoreAttack(attack):
    """Membership inference attack based on WGAN critic scores."""

    name = "critic_score_attack"

    def __init__(
        self,
        n_repeats: int = 5,
        batch_size: int = 128,
        seed: int = 42,
        pack_mode: str = "reference_fillers",
    ) -> None:
        super().__init__()
        self.n_repeats = max(1, int(n_repeats))
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.pack_mode = self._validate_pack_mode(pack_mode)
        self.score_threshold = None
        self.raw_score_threshold = None
        self.raw_score_threshold_0_5 = None
        self.non_train_raw_scores = None
        self.non_member_scores = None
        self.last_raw_scores = None

    def get_display_name(self) -> str:
        return "Critic Score Attack"

    def fit(
        self,
        non_train_data: np.ndarray,
        thr: float = 0.5,
        modelWrapper=None,
    ) -> None:
        """Fit the attack by scoring known non-member samples."""
        wrapper = self._get_wrapper(modelWrapper)
        self.modelWrapper = wrapper
        self.threshold = thr
        self.non_train_data = self._as_2d(non_train_data)

        if len(self.non_train_data) < 1:
            raise ValueError("non_train_data must contain at least one sample")

        self.non_train_raw_scores = self._compute_raw_critic_scores(
            self.non_train_data,
        )
        self.non_member_scores = self._raw_scores_to_membership_scores(
            self.non_train_raw_scores,
        )
        threshold_percentile = self._threshold_to_percentile(self.threshold)
        self.score_threshold = np.percentile(
            self.non_member_scores,
            threshold_percentile,
        )
        self.raw_score_threshold = self._score_threshold_to_raw_score(
            self.score_threshold
        )
        self.raw_score_threshold_0_5 = self._score_threshold_to_raw_score(0.5)
        print(
            "Fitted critic score attack "
            f"with raw non-member mean={np.mean(self.non_train_raw_scores):.4f}, "
            f"std={np.std(self.non_train_raw_scores):.4f}, "
            f"score threshold={self.score_threshold:.4f}, "
            f"raw threshold={self.raw_score_threshold:.6f}, "
            f"raw threshold @ 0.5={self.raw_score_threshold_0_5:.6f}"
        )

        self.is_fitted = True
        return

    def _as_2d(self, data):
        data = np.asarray(data)
        if data.ndim == 1:
            return data.reshape(1, -1)
        return data.reshape(data.shape[0], -1)

    def _get_wrapper(self, modelWrapper=None):
        wrapper = modelWrapper if modelWrapper is not None else getattr(self, "modelWrapper", None)
        if wrapper is None:
            raise ValueError("modelWrapper must be provided or set during fit")
        if not hasattr(wrapper, "critic_score") and getattr(wrapper, "critic", None) is None:
            raise ValueError("modelWrapper must expose a WGAN critic")
        return wrapper

    def _threshold_to_percentile(self, threshold):
        if threshold is None:
            raise ValueError("threshold percentile must be provided")
        if 0 <= threshold <= 1:
            return threshold * 100
        if 0 <= threshold <= 100:
            return threshold
        raise ValueError("threshold percentile must be between 0 and 1 or 0 and 100")

    def _validate_pack_mode(self, pack_mode):
        valid_pack_modes = {"reference_fillers", "repeat_candidate"}
        if pack_mode not in valid_pack_modes:
            raise ValueError(
                f"pack_mode must be one of {sorted(valid_pack_modes)}, got {pack_mode!r}"
            )
        return pack_mode

    def _raw_scores_to_membership_scores(self, raw_scores):
        return np.array([
            np.mean(self.non_train_raw_scores <= score)
            for score in raw_scores
        ])

    def _score_threshold_to_raw_score(self, score_threshold):
        if self.non_train_raw_scores is None:
            raise ValueError("Attack must be fitted before converting thresholds")

        percentile = self._threshold_to_percentile(score_threshold)
        return float(np.percentile(self.non_train_raw_scores, percentile))

    def _sample_fillers(self, n_samples):
        if n_samples <= 0:
            return np.empty((0, self.non_train_data.shape[1]))

        indices = self.rng.integers(
            0,
            len(self.non_train_data),
            size=n_samples,
        )
        return self.non_train_data[indices]

    def _build_packs(self, candidates, candidate_position=0):
        pack_m = getattr(self.modelWrapper, "pack_m", 1)
        if pack_m == 1:
            return candidates[:, np.newaxis, :]
        if not 0 <= candidate_position < pack_m:
            raise ValueError("candidate_position must be within the critic pack")

        packs = np.empty(
            (len(candidates), pack_m, candidates.shape[1]),
            dtype=np.float32,
        )
        fillers = self._sample_fillers(len(candidates) * pack_m)
        packs[:, :, :] = fillers.reshape(len(candidates), pack_m, candidates.shape[1])
        packs[:, candidate_position, :] = candidates
        return packs

    def _build_repeated_candidate_packs(self, candidates):
        pack_m = getattr(self.modelWrapper, "pack_m", 1)
        return np.repeat(candidates[:, np.newaxis, :], pack_m, axis=1).astype(
            np.float32,
            copy=False,
        )

    def _score_packed_batch(self, packed_batch):
        wrapper = self._get_wrapper()
        if hasattr(wrapper, "critic_score"):
            return wrapper.critic_score(packed_batch)

        critic = wrapper.critic
        model_device = next(critic.parameters()).device
        with torch.no_grad():
            batch_tensor = torch.as_tensor(
                packed_batch,
                dtype=torch.float32,
                device=model_device,
            )
            batch_scores = critic(batch_tensor).detach().cpu().numpy()
        return batch_scores.reshape(batch_scores.shape[0])

    def _compute_raw_critic_scores(
        self,
        data: np.ndarray,
        pack_mode: Optional[str] = None,
    ) -> np.ndarray:
        """Average critic scores for each sample over repeated reference packs."""
        self._get_wrapper()
        pack_mode = self._validate_pack_mode(pack_mode or self.pack_mode)
        candidates = self._as_2d(data).astype(np.float32, copy=False)

        if len(candidates) == 0:
            raise ValueError("data must contain at least one sample")

        repeated_scores = []
        for _ in range(self.n_repeats):
            scores = []
            for start in range(0, len(candidates), self.batch_size):
                batch = candidates[start:start + self.batch_size]
                pack_m = getattr(self.modelWrapper, "pack_m", 1)

                if pack_mode == "repeat_candidate":
                    packed_batch = self._build_repeated_candidate_packs(batch)
                    scores.append(self._score_packed_batch(packed_batch))
                    continue

                position_scores = []
                for candidate_position in range(pack_m):
                    packed_batch = self._build_packs(batch, candidate_position)
                    position_scores.append(self._score_packed_batch(packed_batch))

                scores.append(np.stack(position_scores, axis=0).mean(axis=0))
            repeated_scores.append(np.concatenate(scores))

        return np.stack(repeated_scores, axis=0).mean(axis=0)

    def is_attack_applicable(self, model) -> bool:
        return (
            model.get_model_architecture() == "WGAN"
            and getattr(model, "critic", None) is not None
        )

    def score(
        self,
        candidates: np.ndarray,
        pack_mode: Optional[str] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Return membership scores where larger means more likely training."""
        if not hasattr(self, "non_train_raw_scores") or self.non_train_raw_scores is None:
            raise ValueError("Attack must be fitted before scoring")

        raw_scores = self._compute_raw_critic_scores(candidates, pack_mode=pack_mode)
        self.last_raw_scores = raw_scores
        return self._raw_scores_to_membership_scores(raw_scores)

    def raw_score(
        self,
        candidates: np.ndarray,
        pack_mode: Optional[str] = None,
    ) -> np.ndarray:
        """Return averaged raw critic scores before percentile calibration."""
        if not hasattr(self, "non_train_raw_scores") or self.non_train_raw_scores is None:
            raise ValueError("Attack must be fitted before scoring")

        return self._compute_raw_critic_scores(candidates, pack_mode=pack_mode)

    def raw_score_diagnostic(
        self,
        train_data: np.ndarray,
        non_train_data: np.ndarray,
        synthetic_data: np.ndarray,
        model_name: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> dict:
        """Summarize raw critic scores for train, non-train, and synthetic samples."""
        train_raw_scores = self.raw_score(train_data)
        non_train_raw_scores = self.raw_score(non_train_data)
        synthetic_raw_scores = self.raw_score(synthetic_data)

        summary = {
            "model": model_name,
            "pack_mode": self.pack_mode,
            "n_train": len(train_raw_scores),
            "n_non_train": len(non_train_raw_scores),
            "n_synthetic": len(synthetic_raw_scores),
            "train_mean": np.mean(train_raw_scores),
            "train_median": np.median(train_raw_scores),
            "train_std": np.std(train_raw_scores),
            "train_min": np.min(train_raw_scores),
            "train_max": np.max(train_raw_scores),
            "non_train_mean": np.mean(non_train_raw_scores),
            "non_train_median": np.median(non_train_raw_scores),
            "non_train_std": np.std(non_train_raw_scores),
            "non_train_min": np.min(non_train_raw_scores),
            "non_train_max": np.max(non_train_raw_scores),
            "synthetic_mean": np.mean(synthetic_raw_scores),
            "synthetic_median": np.median(synthetic_raw_scores),
            "synthetic_std": np.std(synthetic_raw_scores),
            "synthetic_min": np.min(synthetic_raw_scores),
            "synthetic_max": np.max(synthetic_raw_scores),
        }

        if output_dir is not None:
            if model_name is None:
                raise ValueError("model_name must be provided when output_dir is set")
            out_path = Path(output_dir) / f"{model_name}_{self.name}_diagnostic.csv"
            diagnostic_rows = []
            for split_name, split_scores in (
                ("train", train_raw_scores),
                ("non_train", non_train_raw_scores),
                ("synthetic", synthetic_raw_scores),
            ):
                for sample_index, raw_score in enumerate(split_scores):
                    diagnostic_rows.append({
                        "model": model_name,
                        "pack_mode": self.pack_mode,
                        "split": split_name,
                        "sample_index": sample_index,
                        "average_raw_score": raw_score,
                    })
            pd.DataFrame(diagnostic_rows).to_csv(out_path, index=False)
            summary["path"] = out_path

            plot_path = Path(output_dir) / f"{model_name}_{self.name}_diagnostic_distribution.png"
            self._plot_raw_score_distributions(
                train_raw_scores,
                non_train_raw_scores,
                synthetic_raw_scores,
                plot_path,
            )
            summary["plot_path"] = plot_path

        return summary

    def _plot_raw_score_distributions(
        self,
        train_raw_scores: np.ndarray,
        non_train_raw_scores: np.ndarray,
        synthetic_raw_scores: np.ndarray,
        output_path: Path,
    ) -> None:
        """Save overlaid density lines for raw critic score distributions."""
        all_scores = np.concatenate([
            train_raw_scores,
            non_train_raw_scores,
            synthetic_raw_scores,
        ])
        if len(np.unique(all_scores)) < 2:
            bins = 1
        else:
            bins = np.linspace(np.min(all_scores), np.max(all_scores), 101)

        plt.figure(figsize=(8, 5))
        for label, scores, color in (
            ("Train", train_raw_scores, "tab:blue"),
            ("Non-train", non_train_raw_scores, "tab:orange"),
            ("Synthetic", synthetic_raw_scores, "tab:green"),
        ):
            density, edges = np.histogram(scores, bins=bins, density=True)
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

        plt.xlabel("Average Raw Critic Score")
        plt.ylabel("Density")
        plt.title(f"{self.get_display_name()} Distributions ({self.pack_mode})")
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
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        scores = self.score(candidates, **kwargs)
        thr = self.score_threshold if threshold is None else threshold

        if thr is None:
            raise ValueError("No threshold available. Call fit(...) or pass threshold.")

        predictions = scores >= thr
        return predictions, scores
