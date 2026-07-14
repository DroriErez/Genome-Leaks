"""Discriminator-score membership inference attack for AC-GAN models."""

from typing import Any, Optional, Tuple
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from attack import attack


class DiscriminatorScoreAttack(attack):
    """Membership attack using AC-GAN discriminator validity and class confidence."""

    def __init__(
        self,
        discriminator_weight: float = 1.0,
        class_confidence_weight: float = 0.0,
        batch_size: int = 128,
        class_confidence_mode: str = "true_class",
    ) -> None:
        super().__init__()
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if discriminator_weight < 0 or class_confidence_weight < 0:
            raise ValueError("score weights must be non-negative")
        if discriminator_weight == 0 and class_confidence_weight == 0:
            raise ValueError("at least one score weight must be positive")
        valid_class_confidence_modes = {"max", "predicted_class", "true_class"}
        if class_confidence_mode not in valid_class_confidence_modes:
            raise ValueError(
                "class_confidence_mode must be one of "
                f"{sorted(valid_class_confidence_modes)}"
            )

        total_weight = discriminator_weight + class_confidence_weight
        self.discriminator_weight = discriminator_weight / total_weight
        self.class_confidence_weight = class_confidence_weight / total_weight
        self.batch_size = batch_size
        self.class_confidence_mode = class_confidence_mode
        class_mode_suffix = (
            f"_{class_confidence_mode}"
            if self.class_confidence_weight > 0 else ""
        )
        self.name = (
            "discriminator_score_attack"
            f"_d{self.discriminator_weight:.2f}"
            f"_c{self.class_confidence_weight:.2f}"
            f"{class_mode_suffix}"
        ).replace(".", "p")
        self.score_threshold = None
        self.raw_score_threshold = None
        self.raw_score_threshold_0_5 = None
        self.non_train_discriminator_scores = None
        self.non_train_class_confidences = None
        self.non_member_scores = None
        self.last_raw_scores = None
        self.last_discriminator_scores = None
        self.last_class_confidences = None

    def get_display_name(self) -> str:
        if self.class_confidence_weight == 0:
            return f"Discriminator Score Attack (D={self.discriminator_weight:.2f})"
        return (
            "Discriminator Score Attack "
            f"(D={self.discriminator_weight:.2f}, "
            f"C={self.class_confidence_weight:.2f}, "
            f"{self.class_confidence_mode})"
        )

    @property
    def requires_true_labels(self) -> bool:
        return (
            self.class_confidence_weight > 0
            and self.class_confidence_mode == "true_class"
        )

    def fit(
        self,
        non_train_data: np.ndarray,
        thr: float = 0.5,
        modelWrapper=None,
        non_train_labels=None,
    ) -> None:
        wrapper = self._get_wrapper(modelWrapper)
        self.modelWrapper = wrapper
        self.threshold = thr
        self.non_train_data = self._as_2d(non_train_data)
        self.non_train_labels = self._as_class_ids(
            non_train_labels,
            len(self.non_train_data),
            "non_train_labels",
        )
        if len(self.non_train_data) < 1:
            raise ValueError("non_train_data must contain at least one sample")

        (
            self.non_train_discriminator_scores,
            self.non_train_class_confidences,
        ) = self._compute_raw_components(
            self.non_train_data,
            class_ids=self.non_train_labels,
        )
        self.non_member_scores = self._components_to_membership_scores(
            self.non_train_discriminator_scores,
            self.non_train_class_confidences,
        )
        threshold_percentile = self._threshold_to_percentile(self.threshold)
        self.score_threshold = np.percentile(
            self.non_member_scores,
            threshold_percentile,
        )
        self.raw_score_threshold = self.score_threshold
        self.raw_score_threshold_0_5 = 0.5
        print(
            "Fitted discriminator score attack "
            f"(disc_weight={self.discriminator_weight:.2f}, "
            f"class_weight={self.class_confidence_weight:.2f}, "
            f"class_mode={self.class_confidence_mode}) "
            f"with discriminator mean={np.mean(self.non_train_discriminator_scores):.6f}, "
            f"class_conf_mean={np.mean(self.non_train_class_confidences):.6f}, "
            f"score threshold={self.score_threshold:.4f}"
        )
        self.is_fitted = True

    def _get_wrapper(self, modelWrapper=None):
        wrapper = modelWrapper if modelWrapper is not None else getattr(self, "modelWrapper", None)
        if wrapper is None:
            raise ValueError("modelWrapper must be provided or set during fit")
        if not hasattr(wrapper, "discriminator_score"):
            raise ValueError("modelWrapper must expose discriminator_score")
        if self.class_confidence_weight > 0 and not hasattr(wrapper, "discriminator_outputs"):
            raise ValueError("modelWrapper must expose discriminator_outputs")
        return wrapper

    @staticmethod
    def _as_2d(data):
        data = np.asarray(data)
        if data.ndim == 1:
            return data.reshape(1, -1)
        return data.reshape(data.shape[0], -1)

    def _as_class_ids(self, class_ids, n_samples, name):
        if class_ids is None:
            if self.requires_true_labels:
                raise ValueError(f"{name} is required for true_class confidence")
            return None
        class_ids = np.asarray(class_ids, dtype=int).reshape(-1)
        if len(class_ids) != n_samples:
            raise ValueError(
                f"{name} length must match samples: {len(class_ids)} != {n_samples}"
            )
        return class_ids

    @staticmethod
    def _threshold_to_percentile(threshold):
        if threshold is None:
            raise ValueError("threshold percentile must be provided")
        if 0 <= threshold <= 1:
            return threshold * 100
        if 0 <= threshold <= 100:
            return threshold
        raise ValueError("threshold percentile must be between 0 and 1 or 0 and 100")

    def _compute_raw_components(self, data: np.ndarray, class_ids=None):
        wrapper = self._get_wrapper()
        data = self._as_2d(data).astype(np.float32, copy=False)
        class_ids = self._as_class_ids(class_ids, len(data), "class_ids")
        discriminator_scores = []
        class_confidences = []
        for start in range(0, len(data), self.batch_size):
            batch = data[start:start + self.batch_size]
            validity, class_scores = wrapper.discriminator_outputs(
                batch,
                batch_size=self.batch_size,
            )
            discriminator_scores.append(np.asarray(validity).reshape(-1))
            if self.class_confidence_weight > 0:
                batch_class_ids = None
                if class_ids is not None:
                    batch_class_ids = class_ids[start:start + len(batch)]
                class_confidences.append(
                    self._class_confidences(class_scores, batch_class_ids)
                )

        discriminator_scores = np.concatenate(discriminator_scores)
        if self.class_confidence_weight > 0:
            class_confidences = np.concatenate(class_confidences)
        else:
            class_confidences = np.zeros_like(discriminator_scores)
        return discriminator_scores, class_confidences

    def _class_confidences(self, class_scores, class_ids=None):
        class_scores = np.asarray(class_scores)
        if self.class_confidence_mode in {"max", "predicted_class"}:
            return np.max(class_scores, axis=1).reshape(-1)
        class_ids = self._as_class_ids(class_ids, len(class_scores), "class_ids")
        if np.any(class_ids < 0) or np.any(class_ids >= class_scores.shape[1]):
            raise ValueError("class_ids contain values outside discriminator class range")
        return class_scores[np.arange(len(class_scores)), class_ids].reshape(-1)

    @staticmethod
    def _percentile_scores(reference_scores, candidate_scores):
        return np.array([
            np.mean(reference_scores <= score)
            for score in candidate_scores
        ])

    def _components_to_membership_scores(self, discriminator_scores, class_confidences):
        discriminator_membership = self._percentile_scores(
            self.non_train_discriminator_scores,
            discriminator_scores,
        )
        if self.class_confidence_weight > 0:
            class_membership = self._percentile_scores(
                self.non_train_class_confidences,
                class_confidences,
            )
        else:
            class_membership = np.zeros_like(discriminator_membership)
        return (
            self.discriminator_weight * discriminator_membership
            + self.class_confidence_weight * class_membership
        )

    def is_attack_applicable(self, model) -> bool:
        return (
            model.get_model_architecture() == "AC-GAN"
            and hasattr(model, "discriminator_score")
        )

    def score(self, candidates: np.ndarray, class_ids=None, **kwargs: Any) -> np.ndarray:
        if self.non_train_discriminator_scores is None:
            raise ValueError("Attack must be fitted before scoring")
        discriminator_scores, class_confidences = self._compute_raw_components(
            candidates,
            class_ids=class_ids,
        )
        scores = self._components_to_membership_scores(
            discriminator_scores,
            class_confidences,
        )
        self.last_discriminator_scores = discriminator_scores
        self.last_class_confidences = class_confidences
        self.last_raw_scores = scores
        return scores

    def score_diagnostic(
        self,
        train_data: np.ndarray,
        non_train_data: np.ndarray,
        train_labels=None,
        non_train_labels=None,
        model_name: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> dict:
        """Summarize and save member/non-member discriminator attack scores."""
        train_scores = self.score(train_data, class_ids=train_labels)
        train_discriminator_scores = self.last_discriminator_scores.copy()
        train_class_confidences = self.last_class_confidences.copy()

        non_train_scores = self.score(non_train_data, class_ids=non_train_labels)
        non_train_discriminator_scores = self.last_discriminator_scores.copy()
        non_train_class_confidences = self.last_class_confidences.copy()

        summary = {
            "model": model_name,
            "discriminator_weight": self.discriminator_weight,
            "class_confidence_weight": self.class_confidence_weight,
            "class_confidence_mode": self.class_confidence_mode,
            "n_train": len(train_scores),
            "n_non_train": len(non_train_scores),
            "train_mean": np.mean(train_scores),
            "train_median": np.median(train_scores),
            "train_std": np.std(train_scores),
            "train_min": np.min(train_scores),
            "train_max": np.max(train_scores),
            "non_train_mean": np.mean(non_train_scores),
            "non_train_median": np.median(non_train_scores),
            "non_train_std": np.std(non_train_scores),
            "non_train_min": np.min(non_train_scores),
            "non_train_max": np.max(non_train_scores),
        }

        if output_dir is not None:
            if model_name is None:
                raise ValueError("model_name must be provided when output_dir is set")

            out_path = Path(output_dir) / f"{model_name}_{self.name}_diagnostic.csv"
            diagnostic_rows = []
            for split_name, scores, discriminator_scores, class_confidences in (
                (
                    "train",
                    train_scores,
                    train_discriminator_scores,
                    train_class_confidences,
                ),
                (
                    "non_train",
                    non_train_scores,
                    non_train_discriminator_scores,
                    non_train_class_confidences,
                ),
            ):
                for sample_index, score in enumerate(scores):
                    diagnostic_rows.append({
                        "model": model_name,
                        "split": split_name,
                        "sample_index": sample_index,
                        "membership_score": score,
                        "discriminator_score": discriminator_scores[sample_index],
                        "class_confidence": class_confidences[sample_index],
                    })
            pd.DataFrame(diagnostic_rows).to_csv(out_path, index=False)
            summary["path"] = out_path

            plot_path = Path(output_dir) / f"{model_name}_{self.name}_diagnostic_distribution.png"
            self._plot_score_distributions(
                train_scores,
                non_train_scores,
                plot_path,
            )
            summary["plot_path"] = plot_path

        return summary

    def _plot_score_distributions(
        self,
        train_scores: np.ndarray,
        non_train_scores: np.ndarray,
        output_path: Path,
    ) -> None:
        """Save overlaid fraction histograms for member/non-member attack scores."""
        all_scores = np.concatenate([train_scores, non_train_scores])
        if len(np.unique(all_scores)) < 2:
            bins = 1
        else:
            bins = np.linspace(np.min(all_scores), np.max(all_scores), 101)

        plt.figure(figsize=(8, 5))
        for label, scores, color in (
            ("Member", train_scores, "tab:blue"),
            ("Non-member", non_train_scores, "tab:orange"),
        ):
            weights = np.ones_like(scores, dtype=np.float64) / len(scores)
            fraction, edges = np.histogram(scores, bins=bins, weights=weights)
            centers = (edges[:-1] + edges[1:]) / 2
            smoothed_fraction = self._smooth_density(fraction)
            plt.plot(
                centers,
                fraction,
                color=color,
                linestyle=":",
                linewidth=1.2,
                alpha=0.65,
            )
            plt.plot(
                centers,
                smoothed_fraction,
                label=f"{label} smoothed",
                color=color,
                linewidth=2.2,
            )

        if self.score_threshold is not None:
            plt.axvline(
                self.score_threshold,
                color="black",
                linestyle="--",
                linewidth=1.5,
                label=f"Threshold {self.score_threshold:.3f}",
            )

        plt.xlabel("Membership Score")
        plt.ylabel("Fraction of samples")
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
        class_ids=None,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        scores = self.score(candidates, class_ids=class_ids, **kwargs)
        thr = self.score_threshold if threshold is None else threshold
        if thr is None:
            raise ValueError("No threshold available. Call fit(...) or pass threshold.")
        predictions = scores >= thr
        return predictions, scores
