"""
Monte Carlo Membership Inference Attack

This attack uses Monte Carlo sampling to estimate whether a candidate
is a member of the training set. It samples from the model and measures
how much generated probability mass falls within an epsilon ball around
each candidate.
"""

from typing import Any, Optional, Tuple
import numpy as np
import gc
import sys
import os
import re
from pathlib import Path

try:
    from scipy.spatial import distance as dist
except ImportError:
    dist = None

try:
    import torch
except ImportError:
    torch = None

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from attack import attack


class MonteCarlo_attack(attack):
    name = "monte_carlo_attack"

    def __init__(
        self,
        n_samples: int = 1000,
        distance_metric: str = "euclidean",
        generation_batch_size: int = 32,
        candidate_batch_size: int = 32,
        d_min_n_samples: Optional[int] = None,
        fit_n_samples: Optional[int] = None,
        synthetic_cache_dir: Optional[str] = None,
        synthetic_cache_batch_size: int = 500,
        synthetic_cache_key: Optional[str] = None,
        distance_backend: str = "auto",
        cleanup_interval: int = 25,
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
            d_min_n_samples: Number of synthetic samples to use when
                calibrating ``d_min`` during fit. Defaults to ``n_samples``.
            fit_n_samples: Backward-compatible alias for ``d_min_n_samples``.
            synthetic_cache_dir: Directory for generated synthetic sample
                caches. If omitted, samples are generated on the fly.
            synthetic_cache_batch_size: Number of synthetic samples to generate
                per cache-writing batch.
            synthetic_cache_key: Optional stable cache name. Defaults to the
                model wrapper name during fit.
            distance_backend: Distance implementation. ``auto`` uses torch for
                euclidean distances when available, otherwise scipy.
            cleanup_interval: Run expensive garbage/CUDA cache cleanup every N
                generated batches instead of after every batch.
        """
        super().__init__()
        self.n_samples = n_samples
        self.distance_metric = distance_metric
        self.generation_batch_size = generation_batch_size
        self.candidate_batch_size = candidate_batch_size
        if d_min_n_samples is None:
            d_min_n_samples = fit_n_samples
        self.d_min_n_samples = n_samples if d_min_n_samples is None else d_min_n_samples
        self.synthetic_cache_dir = Path(synthetic_cache_dir) if synthetic_cache_dir is not None else None
        self.synthetic_cache_batch_size = synthetic_cache_batch_size
        self.synthetic_cache_key = synthetic_cache_key
        self.distance_backend = distance_backend
        self.cleanup_interval = cleanup_interval
        self.synthetic_data = None
        self.synthetic_cache_path = None
        self.ref_distances = None
        self.mean_distance = None
        self.std_distance = None
        self.non_train_distances = None
        self.synthetic_distances = None
        self.combined_distances = None
        self.synthetic_mean_distance = None
        self.synthetic_std_distance = None
        self.score_threshold = None
        self.raw_score_threshold = None
        self.raw_score_threshold_0_5 = None
        self.non_member_scores = None
        self.non_member_raw_scores = None
        self.sorted_non_member_raw_scores = None
        self.last_raw_scores = None
        self._cleanup_counter = 0

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
        elif self.synthetic_cache_dir is not None:
            synthetic_data = self.get_synthetic_data(
                wrapper,
                expected_feature_shape=non_train_data.shape[1:],
            )

        d_min_n_samples = self.d_min_n_samples
        if d_min_n_samples <= 0:
            raise ValueError("d_min_n_samples must be positive")

        score_n_samples = len(synthetic_data) if synthetic_data is not None else self.n_samples

        self.non_train_distances = self._compute_nearest_synthetic_distances(
            non_train_data,
            wrapper,
            synthetic_data=synthetic_data,
            n_samples=d_min_n_samples,
        )
        self.d_min = np.median(self.non_train_distances)

        self.non_member_raw_scores = self._raw_score(
            non_train_data,
            synthetic_data=synthetic_data,
            modelWrapper=wrapper,
            n_samples=score_n_samples,
        )
        self.sorted_non_member_raw_scores = np.sort(self.non_member_raw_scores)
        self.non_member_scores = self._raw_scores_to_percentiles(
            self.non_member_raw_scores
        )

        print(
            f"d_min calibrated with {d_min_n_samples} synthetic samples; "
            f"non-member scores calibrated with {score_n_samples} synthetic samples"
        )
        print(
            "Non-member raw score distribution: "
            f"min={np.min(self.non_member_raw_scores):.6f}, "
            f"median={np.median(self.non_member_raw_scores):.6f}, "
            f"max={np.max(self.non_member_raw_scores):.6f}"
        )
        print(f"Non-member percentile score distribution: mean={np.mean(self.non_member_scores):.4f}, std={np.std(self.non_member_scores):.4f}")
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
            f"Score threshold set to {self.score_threshold:.4f} "
            f"(raw={self.raw_score_threshold:.6f}) based on non-member "
            f"percentile distribution"
        )
        print(
            "Raw score value for fixed percentile threshold 0.5: "
            f"{self.raw_score_threshold_0_5:.6f}"
        )
        
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

    def _cleanup_after_batch(self, force=False):
        self._cleanup_counter += 1
        if not force and self.cleanup_interval > 0 and self._cleanup_counter % self.cleanup_interval != 0:
            return
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _torch_device(self):
        if torch is None:
            return None
        wrapper = getattr(self, "modelWrapper", None)
        device = getattr(wrapper, "device", None)
        if device is not None:
            return device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _use_torch_distances(self):
        if self.distance_backend == "scipy":
            if dist is None:
                raise ImportError("scipy is required when distance_backend='scipy'")
            return False
        if self.distance_backend not in ("auto", "torch"):
            raise ValueError("distance_backend must be 'auto', 'torch', or 'scipy'")
        can_use = torch is not None and self.distance_metric in ("euclidean", "sqeuclidean")
        if self.distance_backend == "torch" and not can_use:
            raise ValueError("torch distance backend currently supports only euclidean distances")
        if not can_use and dist is None:
            raise ImportError(
                "scipy is required for non-euclidean Monte Carlo distances when torch cannot be used"
            )
        return can_use

    def _as_float_tensor(self, data, device):
        if torch.is_tensor(data):
            return data.to(device=device, dtype=torch.float32, non_blocking=True)
        return torch.as_tensor(np.asarray(data), dtype=torch.float32, device=device)

    def _squared_euclidean_torch(self, left, right, device=None):
        device = self._torch_device() if device is None else device
        left_tensor = self._as_float_tensor(left, device)
        right_tensor = self._as_float_tensor(right, device)

        left_norm = torch.sum(left_tensor * left_tensor, dim=1, keepdim=True)
        right_norm = torch.sum(right_tensor * right_tensor, dim=1).unsqueeze(0)
        distances_sq = left_norm + right_norm - 2.0 * (left_tensor @ right_tensor.T)
        return torch.clamp(distances_sq, min=0.0)

    def _nearest_distances(self, left, right):
        if self._use_torch_distances():
            with torch.no_grad():
                distances_sq = self._squared_euclidean_torch(left, right)
                nearest_sq = torch.min(distances_sq, dim=1).values
                if self.distance_metric == "euclidean":
                    nearest = torch.sqrt(nearest_sq)
                else:
                    nearest = nearest_sq
                return nearest.detach().cpu().numpy()

        distances = dist.cdist(left, right, metric=self.distance_metric)
        return np.min(distances, axis=1)

    def _count_inside_d_min(self, left, right):
        if self._use_torch_distances():
            threshold = self.d_min * self.d_min if self.distance_metric == "euclidean" else self.d_min
            with torch.no_grad():
                distances_sq = self._squared_euclidean_torch(left, right)
                return torch.sum(distances_sq < threshold, dim=1).detach().cpu().numpy()

        distances = dist.cdist(left, right, metric=self.distance_metric)
        counts = np.sum(distances < self.d_min, axis=1)
        del distances
        return counts

    def _raw_scores_to_percentiles(self, raw_scores):
        if self.sorted_non_member_raw_scores is None:
            raise ValueError("Attack must be fitted before converting scores to percentiles")

        raw_scores = np.asarray(raw_scores)
        left_ranks = np.searchsorted(
            self.sorted_non_member_raw_scores,
            raw_scores,
            side="left",
        )
        right_ranks = np.searchsorted(
            self.sorted_non_member_raw_scores,
            raw_scores,
            side="right",
        )
        return (left_ranks + right_ranks) / (2 * len(self.sorted_non_member_raw_scores))

    def _score_threshold_to_raw_score(self, score_threshold):
        if self.sorted_non_member_raw_scores is None:
            raise ValueError("Attack must be fitted before converting thresholds")

        percentile = self._threshold_to_percentile(score_threshold)
        return float(np.percentile(self.sorted_non_member_raw_scores, percentile))

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

    def _safe_cache_name(self, value):
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "model"

    def get_synthetic_data(self, modelWrapper=None, expected_feature_shape=None):
        wrapper = self._get_wrapper(modelWrapper)
        if self.synthetic_data is not None:
            return self.synthetic_data

        if self.synthetic_cache_dir is None:
            self.synthetic_data = self._generate_samples(wrapper, self.n_samples)
            return self.synthetic_data

        self.synthetic_data, self.synthetic_cache_path = self._load_or_create_synthetic_cache(
            wrapper,
            self.n_samples,
            expected_feature_shape=expected_feature_shape,
        )
        return self.synthetic_data

    def _load_or_create_synthetic_cache(
        self,
        wrapper,
        n_samples,
        expected_feature_shape=None,
    ):
        if n_samples < 1:
            raise ValueError("n_samples must be at least 1")
        if self.synthetic_cache_batch_size < 1:
            raise ValueError("synthetic_cache_batch_size must be at least 1")

        cache_dir = Path(self.synthetic_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = self.synthetic_cache_key or getattr(wrapper, "model_name", "model")
        cache_name = self._safe_cache_name(cache_key)
        cache_path = cache_dir / f"{cache_name}_synthetic_{n_samples}.npy"

        if cache_path.exists():
            cached = np.load(cache_path, mmap_mode="r")
            shape_matches = (
                expected_feature_shape is None
                or tuple(cached.shape[1:]) == tuple(expected_feature_shape)
            )
            if cached.shape[0] >= n_samples and shape_matches:
                print(f"Using cached synthetic samples: {cache_path}")
                return cached[:n_samples], cache_path
            print(
                f"Ignoring cached synthetic samples with shape {cached.shape}; "
                f"expected ({n_samples}, {expected_feature_shape}). Regenerating."
            )

        temp_path = cache_path.with_suffix(".tmp.npy")
        if temp_path.exists():
            temp_path.unlink()

        first_batch_size = min(self.synthetic_cache_batch_size, n_samples)
        first_batch = self._as_2d(wrapper.generate(n=first_batch_size))
        if len(first_batch) == 0:
            raise ValueError("modelWrapper.generate(...) returned an empty batch")
        first_batch = first_batch[:first_batch_size]

        feature_shape = first_batch.shape[1:]
        if expected_feature_shape is not None and tuple(feature_shape) != tuple(expected_feature_shape):
            raise ValueError(
                f"Generated synthetic samples have feature shape {feature_shape}, "
                f"expected {expected_feature_shape}"
            )

        synthetic_memmap = np.lib.format.open_memmap(
            temp_path,
            mode="w+",
            dtype=first_batch.dtype,
            shape=(n_samples, *feature_shape),
        )
        synthetic_memmap[:len(first_batch)] = first_batch[:n_samples]
        generated_so_far = len(first_batch)
        print(f"Generated synthetic cache batch: {generated_so_far}/{n_samples}")

        while generated_so_far < n_samples:
            current_batch_size = min(
                self.synthetic_cache_batch_size,
                n_samples - generated_so_far,
            )
            batch = self._as_2d(wrapper.generate(n=current_batch_size))
            if len(batch) == 0:
                raise ValueError("modelWrapper.generate(...) returned an empty batch")
            batch = batch[:current_batch_size]
            if batch.shape[1:] != feature_shape:
                raise ValueError(
                    f"Generated batch feature shape {batch.shape[1:]} does not match {feature_shape}"
                )

            end = generated_so_far + len(batch)
            synthetic_memmap[generated_so_far:end] = batch
            generated_so_far = end
            print(f"Generated synthetic cache batch: {generated_so_far}/{n_samples}")

        synthetic_memmap.flush()
        del synthetic_memmap
        temp_path.replace(cache_path)
        print(f"Saved synthetic samples cache: {cache_path}")
        return np.load(cache_path, mmap_mode="r"), cache_path

    def _get_synthetic_batch(self, start, batch_size, synthetic_data):
        indices = (np.arange(batch_size) + start) % len(synthetic_data)
        return synthetic_data[indices]

    def _compute_nearest_synthetic_distances(
        self,
        non_train_data,
        wrapper,
        synthetic_data=None,
        n_samples=None,
    ):
        n_samples = self.n_samples if n_samples is None else n_samples
        nearest_distances = np.full(len(non_train_data), np.inf)

        if synthetic_data is None:
            generated_so_far = 0
            while generated_so_far < n_samples:
                current_batch_size = min(
                    self.generation_batch_size,
                    n_samples - generated_so_far,
                )
                synthetic_batch = self._as_2d(wrapper.generate(current_batch_size))

                if synthetic_batch.shape[0] == 0:
                    raise ValueError("modelWrapper.generate(...) returned an empty batch")

                nearest_distances = np.minimum(
                    nearest_distances,
                    self._nearest_distances(non_train_data, synthetic_batch),
                )
                generated_so_far += synthetic_batch.shape[0]
                del synthetic_batch
                self._cleanup_after_batch()
        else:
            usable_n_samples = min(n_samples, len(synthetic_data))
            for start in range(0, usable_n_samples, self.generation_batch_size):
                end = min(start + self.generation_batch_size, usable_n_samples)
                synthetic_batch = synthetic_data[start:end]
                nearest_distances = np.minimum(
                    nearest_distances,
                    self._nearest_distances(non_train_data, synthetic_batch),
                )
                self._cleanup_after_batch()

        return nearest_distances

    def _raw_score(self, candidates, synthetic_data=None, modelWrapper=None, **kwargs):
        """
        Estimate raw MC-epsilon membership scores by generating samples in batches.

        For each candidate x, the score is:
            mean_j 1[d(x, g_j) < epsilon]

        where g_j are generated on the fly from ``modelWrapper`` and epsilon is
        the median nearest-synthetic distance computed during ``fit``.
        """
        if not hasattr(self, "d_min"):
            raise ValueError("Attack must be fitted before scoring")

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

        wrapper = None
        if synthetic_data is None:
            if self.synthetic_data is not None:
                synthetic_data = self.synthetic_data
            elif self.synthetic_cache_dir is not None:
                synthetic_data = self.get_synthetic_data(modelWrapper)
            else:
                wrapper = self._get_wrapper(modelWrapper)

        if synthetic_data is not None:
            synthetic_data = self._as_2d(synthetic_data)
            if len(synthetic_data) == 0:
                raise ValueError("synthetic_data must contain at least 1 sample")

        counts = np.zeros(len(candidates), dtype=float)
        generated_so_far = 0

        while generated_so_far < n_samples:
            current_batch_size = min(generation_batch_size, n_samples - generated_so_far)
            if synthetic_data is None:
                generated_batch = self._as_2d(wrapper.generate(current_batch_size))
            else:
                generated_batch = self._as_2d(
                    self._get_synthetic_batch(
                        generated_so_far,
                        current_batch_size,
                        synthetic_data,
                    )
                )

            if generated_batch.shape[0] == 0:
                raise ValueError("modelWrapper.generate(...) returned an empty batch")

            generated_count = generated_batch.shape[0]
            for start in range(0, len(candidates), candidate_batch_size):
                end = min(start + candidate_batch_size, len(candidates))
                counts[start:end] += self._count_inside_d_min(
                    candidates[start:end],
                    generated_batch,
                )

            del generated_batch
            generated_so_far += generated_count
            self._cleanup_after_batch()

        return counts / generated_so_far

    def score(self, candidates, synthetic_data=None, modelWrapper=None, **kwargs):
        """
        Return percentile membership scores in [0, 1].

        The raw Monte Carlo score is compared against the fitted evaluation
        dataset score distribution. A larger percentile means the candidate is
        more train-like relative to known non-member/evaluation samples.
        """
        raw_scores = self._raw_score(
            candidates,
            synthetic_data=synthetic_data,
            modelWrapper=modelWrapper,
            **kwargs,
        )
        self.last_raw_scores = raw_scores
        return self._raw_scores_to_percentiles(raw_scores)
    
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
