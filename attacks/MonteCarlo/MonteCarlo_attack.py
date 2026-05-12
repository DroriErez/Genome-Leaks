"""
Monte Carlo Membership Inference Attack

This attack uses Monte Carlo sampling to estimate whether a candidate
is a member of the training set. It samples from the model and measures
how much generated probability mass falls within an epsilon ball around
each candidate.
"""

from typing import Any, Optional, Tuple
import numpy as np
from scipy.spatial import distance as dist
import gc
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from attack import attack


class MonteCarlo_attack(attack):
    name = "monte_carlo_attack"

    def __init__(
        self,
        n_samples: int = 1000,
        distance_metric: str = "euclidean",
        generation_batch_size: int = 128,
        candidate_batch_size: int = 256,
    ) -> None:
        """
        Initialize Monte Carlo Attack.

        Args:
            n_samples: Number of Monte Carlo samples to draw when scoring
            distance_metric: Distance metric to use ('euclidean', 'cosine', etc.)
            generation_batch_size: Number of synthetic samples to generate per
                modelWrapper.generate(...) call during scoring.
            candidate_batch_size: Number of candidates to compare per distance
                call. Lower this if scoring raises a memory exception.
        """
        super().__init__()
        self.n_samples = n_samples
        self.distance_metric = distance_metric
        self.generation_batch_size = generation_batch_size
        self.candidate_batch_size = candidate_batch_size
        self.ref_distances = None
        self.mean_distance = None
        self.std_distance = None
        self.non_train_distances = None
        self.synthetic_distances = None
        self.combined_distances = None
        self.synthetic_mean_distance = None
        self.synthetic_std_distance = None
        self.score_threshold = None
        self.non_member_scores = None

    def fit(
        self,
        non_train_data: np.ndarray,
        synthetic_data: Optional[np.ndarray] = None,
        thr: float = 0.5,
        modelWrapper=None,
    ) -> None:
        """
        Fit the attack and calibrate its decision threshold.

        ``thr`` is treated as a percentile of the non-member score
        distribution. For example, ``thr=0.95`` sets the prediction cutoff to
        the 95th percentile of scores measured on known non-training records.
        """
        self.modelWrapper = modelWrapper
        self.threshold = thr
        self._fit(
            non_train_data=non_train_data,
            synthetic_data=synthetic_data,
            modelWrapper=modelWrapper,
        )
        self.is_fitted = True
        return

    def _fit(self, non_train_data, synthetic_data=None, modelWrapper=None):
        wrapper = self._get_wrapper(modelWrapper)
        non_train_data = self._as_2d(non_train_data)

        if len(non_train_data) < 1:
            raise ValueError("non_train_data must contain at least 1 sample")

        if synthetic_data is not None:
            synthetic_data = self._as_2d(synthetic_data)
            if len(synthetic_data) < 2:
                raise ValueError("Synthetic data must contain at least 2 samples")

        self.non_train_distances = self._compute_nearest_synthetic_distances(
            non_train_data,
            wrapper,
            synthetic_data=synthetic_data,
        )
        self.d_min = np.median(self.non_train_distances)

        self.non_member_scores = self.score(
            non_train_data,
            modelWrapper=wrapper,
        )

        print(f"Non-member score distribution: mean={np.mean(self.non_member_scores):.4f}, std={np.std(self.non_member_scores):.4f}")
        self.score_threshold = np.percentile(
            self.non_member_scores,
            self._threshold_to_percentile(self.threshold),
        )
        print(f"Score threshold set to {self.score_threshold:.4f} based on non-member distribution")
        
    def _as_2d(self, data):
        data = np.asarray(data)
        if data.ndim == 1:
            return data.reshape(1, -1)
        return data.reshape(data.shape[0], -1)

    def _get_wrapper(self, modelWrapper=None):
        wrapper = modelWrapper if modelWrapper is not None else self.modelWrapper
        if wrapper is None:
            raise ValueError("modelWrapper must be provided or set during fit")
        if not hasattr(wrapper, "generate"):
            raise ValueError("modelWrapper must expose a generate(n) method")
        return wrapper

    def _threshold_to_percentile(self, threshold):
        if threshold is None:
            raise ValueError("threshold percentile must be provided")
        if 0 <= threshold <= 1:
            return threshold * 100
        if 0 <= threshold <= 100:
            return threshold
        raise ValueError("threshold percentile must be between 0 and 1 or 0 and 100")

    def _generate_samples(self, wrapper, n_samples):
        generated = []
        generated_so_far = 0

        while generated_so_far < n_samples:
            current_batch_size = min(
                self.generation_batch_size,
                n_samples - generated_so_far,
            )
            generated_batch = self._as_2d(wrapper.generate(current_batch_size))

            if generated_batch.shape[0] == 0:
                raise ValueError("modelWrapper.generate(...) returned an empty batch")

            generated.append(generated_batch)
            generated_so_far += generated_batch.shape[0]

        return np.vstack(generated)

    def _get_synthetic_batch(self, start, batch_size, synthetic_data):
        indices = (np.arange(batch_size) + start) % len(synthetic_data)
        return synthetic_data[indices]

    def _compute_nearest_synthetic_distances(
        self,
        non_train_data,
        wrapper,
        synthetic_data=None,
    ):
        nearest_distances = np.full(len(non_train_data), np.inf)

        if synthetic_data is None:
            generated_so_far = 0
            while generated_so_far < self.n_samples:
                current_batch_size = min(
                    self.generation_batch_size,
                    self.n_samples - generated_so_far,
                )
                synthetic_batch = self._as_2d(wrapper.generate(current_batch_size))

                if synthetic_batch.shape[0] == 0:
                    raise ValueError("modelWrapper.generate(...) returned an empty batch")

                distances = dist.cdist(
                    non_train_data,
                    synthetic_batch,
                    metric=self.distance_metric,
                )
                nearest_distances = np.minimum(
                    nearest_distances,
                    np.min(distances, axis=1),
                )
                generated_so_far += synthetic_batch.shape[0]
                del synthetic_batch, distances
                gc.collect()
        else:
            for start in range(0, len(synthetic_data), self.generation_batch_size):
                synthetic_batch = synthetic_data[start:start + self.generation_batch_size]
                distances = dist.cdist(
                    non_train_data,
                    synthetic_batch,
                    metric=self.distance_metric,
                )
                nearest_distances = np.minimum(
                    nearest_distances,
                    np.min(distances, axis=1),
                )
                del distances
                gc.collect()

        return nearest_distances

    def score(self, candidates, synthetic_data=None, modelWrapper=None, **kwargs):
        """
        Estimate MC-epsilon membership scores by generating samples in batches.

        For each candidate x, the score is:
            mean_j 1[d(x, g_j) < epsilon]

        where g_j are generated on the fly from ``modelWrapper`` and epsilon is
        the median nearest-synthetic distance computed during ``fit``.
        """
        if not hasattr(self, "d_min"):
            raise ValueError("Attack must be fitted before scoring")

        wrapper = self._get_wrapper(modelWrapper)

        n_samples = kwargs.get("n_samples", self.n_samples)
        generation_batch_size = kwargs.get(
            "generation_batch_size", self.generation_batch_size
        )
        candidate_batch_size = kwargs.get(
            "candidate_batch_size", self.candidate_batch_size
        )

        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if generation_batch_size <= 0:
            raise ValueError("generation_batch_size must be positive")
        if candidate_batch_size <= 0:
            raise ValueError("candidate_batch_size must be positive")

        candidates = self._as_2d(candidates)
        if len(candidates) == 0:
            raise ValueError("candidates must contain at least 1 sample")

        counts = np.zeros(len(candidates), dtype=float)
        generated_so_far = 0

        while generated_so_far < n_samples:
            current_batch_size = min(generation_batch_size, n_samples - generated_so_far)
            generated_batch = self._as_2d(wrapper.generate(current_batch_size))
            # generated_batch = self._get_synthetic_batch(
            #     synthetic_start,
            #     current_batch_size,
            #     synthetic_data,
            # )

            if generated_batch.shape[0] == 0:
                raise ValueError("modelWrapper.generate(...) returned an empty batch")

            generated_count = generated_batch.shape[0]
            for start in range(0, len(candidates), candidate_batch_size):
                end = min(start + candidate_batch_size, len(candidates))
                distances = dist.cdist(
                    candidates[start:end],
                    generated_batch,
                    metric=self.distance_metric,
                )
                counts[start:end] += np.sum(distances < self.d_min, axis=1)

            del generated_batch, distances
            generated_so_far += generated_count
            gc.collect()

        return counts / generated_so_far
    
    def predict(
        self,
        candidates: np.ndarray,
        reference_data: Optional[np.ndarray] = None,
        synthetic_data: Optional[np.ndarray] = None,
        threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict membership of candidates using Monte Carlo attack.

        Args:
            candidates: Candidate samples to test
            reference_data: Not used (kept for compatibility)
            synthetic_data: Synthetic data to compare against
            threshold: Decision threshold (uses self.threshold if None)
            **kwargs: Additional arguments

        Returns:
            Tuple of (predictions, scores) where predictions are boolean
        """
        scores = self.score(candidates, synthetic_data=synthetic_data, **kwargs)
        thr = self.score_threshold if threshold is None else threshold

        if thr is None:
            raise ValueError(
                "No threshold available. Call fit(...) or pass threshold."
            )

        predictions = scores >= thr
        return predictions, scores
