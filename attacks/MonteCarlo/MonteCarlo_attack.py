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
        distance_backend: str = "auto",
        cleanup_interval: int = 25,
        neighborhood: str = "epsilon",
        score_function: str = "count",
        k_neighbors: int = 1,
        log_delta: float = 1e-12,
        pca_components: int = 50,
        min_heterozygosity: float = 0.01,
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
            distance_backend: Distance implementation. ``auto`` uses torch for
                euclidean distances when available, otherwise scipy.
            cleanup_interval: Run expensive garbage/CUDA cache cleanup every N
                generated batches instead of after every batch.
            neighborhood: Select distances inside ``d_min`` (``epsilon``) or
                the ``k`` nearest generated samples (``knn``).
            score_function: Aggregate selected distances using ``count``,
                ``log_distance``, or ``average_distance``. Distance-based raw
                scores are negated so that larger always means more member-like.
            k_neighbors: Number of nearest samples used by ``knn``.
            log_delta: Positive numerical-stability constant added before log.
        """
        super().__init__()
        self.n_samples = n_samples
        self.distance_metric = distance_metric
        self.allele_frequency = None
        if distance_metric == "allele_similarity" and (
                neighborhood != "knn" or score_function != "average_distance"):
            raise ValueError("Allele similarity requires KNN average scoring")
        if pca_components < 1 or min_heterozygosity <= 0:
            raise ValueError("PCA components and minimum heterozygosity must be positive")
        self.pca_components = pca_components
        self.min_heterozygosity = min_heterozygosity
        self.pca = None
        self.feature_weights = None
        self.generation_batch_size = generation_batch_size
        self.candidate_batch_size = candidate_batch_size
        if d_min_n_samples is None:
            d_min_n_samples = fit_n_samples
        self.d_min_n_samples = n_samples if d_min_n_samples is None else d_min_n_samples
        self.distance_backend = distance_backend
        self.cleanup_interval = cleanup_interval
        if neighborhood not in ("epsilon", "knn"):
            raise ValueError("neighborhood must be 'epsilon' or 'knn'")
        if score_function not in ("count", "log_distance", "average_distance"):
            raise ValueError(
                "score_function must be 'count', 'log_distance', or "
                "'average_distance'"
            )
        if neighborhood == "knn" and score_function == "count":
            raise ValueError(
                "count scoring is constant for a fixed k; use log_distance "
                "or average_distance with neighborhood='knn'"
            )
        if k_neighbors <= 0:
            raise ValueError("k_neighbors must be positive")
        if log_delta <= 0:
            raise ValueError("log_delta must be positive")
        self.neighborhood = neighborhood
        self.score_function = score_function
        self.k_neighbors = k_neighbors
        self.log_delta = log_delta
        neighborhood_name = "epsilon" if neighborhood == "epsilon" else f"knn_k{k_neighbors}"
        self.name = f"monte_carlo_{neighborhood_name}_{score_function}"
        if distance_metric in ("allele_similarity", "pca_euclidean", "weighted_hamming", "hamming", "jaccard", "weighted_jaccard"):
            self.name += f"_{distance_metric}"
        if distance_metric == "pca_euclidean" and pca_components == 2:
            self.name += "_pca2"
        self.synthetic_data = None
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

    def shared_knn_signals(self, attacks, candidates, synthetic_data, modelWrapper,
                           frequency_reference=None):
        """Evaluate compatible KNN configurations in one bounded-memory scan.

        For PCA distances, fit the full-cache PCA and project each batch once.
        Hamming uses the original features. Snapshot each requested prefix.
        """
        for other in attacks:
            if (self.distance_metric not in ("pca_euclidean", "hamming", "weighted_hamming", "allele_similarity")
                    or other.distance_metric != self.distance_metric
                    or other.neighborhood != "knn"
                    or other.score_function != "average_distance"
                    or other.pca_components != self.pca_components
                    or other.distance_backend != self.distance_backend
                    or other.min_heterozygosity != self.min_heterozygosity):
                raise ValueError("Shared KNN sweep requires compatible attacks")
            if not 0 < other.k_neighbors <= other.n_samples <= len(synthetic_data):
                raise ValueError("Shared sweep requires k <= sample count <= cache size")
        if self.generation_batch_size < 1 or self.candidate_batch_size < 1:
            raise ValueError("Shared sweep batch sizes must be positive")
        if self.distance_metric == "pca_euclidean":
            print("Fitting shared PCA on the synthetic cache...", flush=True)
        self.prepare(synthetic_data=synthetic_data, modelWrapper=modelWrapper,
                     frequency_reference=frequency_reference)
        projected_candidates = self._as_2d(candidates)
        if self.distance_metric == "pca_euclidean":
            projected_candidates = self.pca.transform(projected_candidates)
        max_k = max(a.k_neighbors for a in attacks)
        nearest = np.full((len(candidates), max_k), np.inf)
        signals = {}
        use_torch = self._use_torch_distances()
        start = 0
        for boundary in sorted({a.n_samples for a in attacks}):
            while start < boundary:
                end = min(start + self.generation_batch_size, boundary)
                projected = self._as_2d(synthetic_data[start:end])
                if self.distance_metric == "pca_euclidean":
                    projected = self.pca.transform(projected)
                for row in range(0, len(candidates), self.candidate_batch_size):
                    stop = min(row + self.candidate_batch_size, len(candidates))
                    left = projected_candidates[row:stop]
                    if self.distance_metric in ("weighted_hamming", "allele_similarity"):
                        distances = self._pairwise_distances(left, projected)
                    elif use_torch:
                        with torch.no_grad():
                            if self.distance_metric == "hamming":
                                distances = self._normalized_hamming_torch(left, projected).cpu().numpy()
                            else:
                                distances = torch.sqrt(
                                    self._squared_euclidean_torch(left, projected)
                                ).cpu().numpy()
                    else:
                        metric = "hamming" if self.distance_metric == "hamming" else "euclidean"
                        distances = dist.cdist(left, projected, metric=metric)
                    combined = np.concatenate([nearest[row:stop], distances], axis=1)
                    nearest[row:stop] = np.partition(
                        combined, max_k - 1, axis=1
                    )[:, :max_k]
                start = end
            ordered = np.sort(nearest, axis=1)
            for other in attacks:
                if other.n_samples == boundary:
                    signals[other.name] = -np.mean(ordered[:, :other.k_neighbors], axis=1)
            print(f"Shared {self.distance_metric} KNN sweep: {boundary} synthetic samples evaluated", flush=True)
        return signals

    def get_display_name(self) -> str:
        neighborhood = (
            "epsilon"
            if self.neighborhood == "epsilon"
            else f"{self.k_neighbors}-nearest"
        )
        score = self.score_function.replace("_", " ")
        metric_label = self.distance_metric
        if self.distance_metric == "pca_euclidean":
            metric_label += f", {self.pca_components} components"
        return f"Monte Carlo ({neighborhood}, {score}, {metric_label})"

    def fit(
        self,
        non_train_data: np.ndarray,
        synthetic_data: Optional[np.ndarray] = None,
        thr: float = 0.5,
        modelWrapper=None,
        training_reference=None,
        frequency_reference=None,
    ) -> None:
        """
        Fit the attack and calibrate its decision threshold.

        ``thr`` is treated as a percentile of the non-member score
        distribution. For example, ``thr=0.95`` sets the prediction cutoff to
        the 95th percentile of scores measured on known non-training records.
        Weighted Hamming and weighted Jaccard use these calibration records for allele frequencies
        unless an explicit frequency_reference is supplied. training_reference
        is retained as a legacy override for existing callers.
        """
        self.prepare(
            non_train_data, synthetic_data, modelWrapper,
            training_reference=training_reference,
            frequency_reference=frequency_reference,
        )
        self.threshold = thr
        self._fit(
            non_train_data=non_train_data,
            synthetic_data=synthetic_data,
            modelWrapper=modelWrapper,
        )
        self.is_fitted = True
        return

    def prepare(
        self, non_train_data=None, synthetic_data=None, modelWrapper=None,
        training_reference=None, frequency_reference=None,
    ):
        """Prepare distance features and epsilon without decision calibration.

        Epsilon methods still estimate their neighborhood radius from reference
        data. KNN methods need no radius or non-member score distribution.
        """
        self.is_fitted = False
        self.score_threshold = None
        self.raw_score_threshold = None
        self.raw_score_threshold_0_5 = None
        self.non_member_scores = None
        self.non_member_raw_scores = None
        self.sorted_non_member_raw_scores = None
        self.last_raw_scores = None
        self.synthetic_data = synthetic_data
        self.modelWrapper = modelWrapper
        if self.distance_metric == "allele_similarity":
            reference = frequency_reference if frequency_reference is not None else non_train_data
            if reference is None:
                raise ValueError("Allele similarity requires a frequency reference")
            reference = self._as_2d(reference)
            if len(reference) == 0 or not np.all((reference == 0) | (reference == 1)):
                raise ValueError("Allele similarity requires nonempty binary reference genomes")
            # Clip monomorphic frequencies to keep logarithms finite.
            self.allele_frequency = np.clip(reference.mean(axis=0, dtype=np.float64),
                                            1e-6, 1.0 - 1e-6)
        elif self.distance_metric in ("weighted_hamming", "weighted_jaccard"):
            if frequency_reference is None:
                frequency_reference = non_train_data if training_reference is None else training_reference
            reference = self._as_2d(frequency_reference)
            if not np.all((reference == 0) | (reference == 1)) or len(reference) == 0:
                raise ValueError("Weighted distances require nonempty binary frequency-reference genomes")
            frequency = np.mean(reference, axis=0, dtype=np.float64)
            self.feature_weights = 1.0 / np.maximum(
                2.0 * frequency * (1.0 - frequency), self.min_heterozygosity
            )
        elif self.distance_metric == "pca_euclidean":
            from sklearn.decomposition import IncrementalPCA
            if synthetic_data is None:
                raise ValueError("PCA Euclidean requires a shared synthetic reference during fit")
            reference = self._as_2d(synthetic_data)
            if len(reference) < 2:
                raise ValueError("PCA requires at least two synthetic samples")
            n_components = min(
                self.pca_components, len(reference) - 1, reference.shape[1]
            )
            # Convert only a bounded slice of the memory-mapped cache. Fitting
            # ordinary PCA on integer genomes converts the entire cache to
            # float64 (12.2 GiB for 100000 x 16383, before working memory).
            pca_batch_size = max(n_components, min(self.generation_batch_size, 512))
            self.pca = IncrementalPCA(n_components=n_components)
            start = 0
            while start < len(reference):
                end = min(start + pca_batch_size, len(reference))
                # partial_fit requires at least n_components rows. Fold a short
                # final remainder into this batch so every sample is used once.
                if 0 < len(reference) - end < n_components:
                    end = len(reference)
                batch = np.array(reference[start:end], dtype=np.float32, copy=True)
                self.pca.partial_fit(batch)
                del batch
                start = end
        if self.neighborhood == "epsilon":
            if non_train_data is None or len(non_train_data) == 0:
                raise ValueError("Epsilon preparation requires non-member reference data")
            if self.d_min_n_samples <= 0:
                raise ValueError("d_min_n_samples must be positive")
            self.non_train_distances = self._compute_nearest_synthetic_distances(
                self._as_2d(non_train_data), self._get_wrapper(modelWrapper),
                synthetic_data=synthetic_data, n_samples=self.d_min_n_samples,
            )
            self.d_min = np.median(self.non_train_distances)

    def raw_signal(self, candidates, synthetic_data=None, modelWrapper=None, **kwargs):
        """Return uncalibrated signals, oriented so larger means more member-like."""
        return self._raw_score(candidates, synthetic_data, modelWrapper, **kwargs)

    def execute(self, candidates, **kwargs):
        """Execute calibrated membership decisions after fit()."""
        return self.predict(candidates, **kwargs)

    def _fit(self, non_train_data, synthetic_data=None, modelWrapper=None):
        wrapper = self._get_wrapper(modelWrapper)
        non_train_data = self._as_2d(non_train_data)

        if len(non_train_data) < 1:
            raise ValueError("non_train_data must contain at least 1 sample")

        if synthetic_data is not None:
            synthetic_data = self._as_2d(synthetic_data)
            if len(synthetic_data) < 2:
                raise ValueError("Synthetic data must contain at least 2 samples")
            self.synthetic_data = synthetic_data

        d_min_n_samples = self.d_min_n_samples
        if d_min_n_samples <= 0:
            raise ValueError("d_min_n_samples must be positive")

        score_n_samples = len(synthetic_data) if synthetic_data is not None else self.n_samples

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

        if self.neighborhood == "epsilon":
            print(f"d_min estimated with {d_min_n_samples} synthetic samples")
        print(f"Non-member scores calibrated with {score_n_samples} synthetic samples")
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
        can_use = torch is not None and self.distance_metric in (
            "allele_similarity",
            "euclidean",
            "sqeuclidean",
            "hamming",
            "pca_euclidean",
            "weighted_hamming",
            "jaccard",
            "weighted_jaccard",
        )
        if self.distance_backend == "torch" and not can_use:
            raise ValueError(
                "torch distance backend currently supports euclidean, "
                "squared-euclidean, Hamming, and Jaccard distances"
            )
        if not can_use and dist is None:
            raise ImportError(
                "scipy is required for non-euclidean Monte Carlo distances when torch cannot be used"
            )
        return can_use

    def _as_float_tensor(self, data, device):
        if torch.is_tensor(data):
            return data.to(device=device, dtype=torch.float32, non_blocking=True)
        array = np.asarray(data)
        if not array.flags.writeable:
            # Synthetic caches are read-only memory maps. PyTorch warns when
            # sharing their storage because tensor writes would be undefined.
            # Copy only the current batch, not the full cached dataset.
            array = array.copy()
        return torch.as_tensor(array, dtype=torch.float32, device=device)

    def _squared_euclidean_torch(self, left, right, device=None):
        device = self._torch_device() if device is None else device
        left_tensor = self._as_float_tensor(left, device)
        right_tensor = self._as_float_tensor(right, device)

        left_norm = torch.sum(left_tensor * left_tensor, dim=1, keepdim=True)
        right_norm = torch.sum(right_tensor * right_tensor, dim=1).unsqueeze(0)
        distances_sq = left_norm + right_norm - 2.0 * (left_tensor @ right_tensor.T)
        return torch.clamp(distances_sq, min=0.0)

    def _normalized_hamming_torch(self, left, right, device=None):
        """Return the fraction of unequal features for every pair of rows."""
        device = self._torch_device() if device is None else device
        left_tensor = self._as_float_tensor(left, device)
        right_tensor = self._as_float_tensor(right, device)
        return torch.cdist(left_tensor, right_tensor, p=0) / left_tensor.shape[1]

    def _nearest_distances(self, left, right):
        distances = self._pairwise_distances(left, right)
        return np.min(distances, axis=1)

    def _pairwise_distances(self, left, right):
        """Return a CPU NumPy distance matrix, using torch/GPU when available."""
        if self.distance_metric == "allele_similarity":
            if self.allele_frequency is None:
                raise ValueError("Prepare allele frequencies before computing similarity")
            if not (np.all((left == 0) | (left == 1))
                    and np.all((right == 0) | (right == 1))):
                raise ValueError("Allele similarity requires binary genomes")
            log_p = np.log(self.allele_frequency)
            log_q = np.log1p(-self.allele_frequency)
            # Return -similarity: the KNN machinery minimizes distances and
            # negates their mean, yielding the mean of the largest similarities.
            if self._use_torch_distances():
                with torch.no_grad():
                    device = self._torch_device()
                    x = self._as_float_tensor(left, device)
                    y = self._as_float_tensor(right, device)
                    lp = self._as_float_tensor(log_p, device)
                    lq = self._as_float_tensor(log_q, device)
                    return (((x * lp) @ y.T + ((1 - x) * lq) @ (1 - y).T)
                            / x.shape[1]).cpu().numpy()
            x = np.asarray(left, dtype=np.float64)
            y = np.asarray(right, dtype=np.float64)
            return ((x * log_p) @ y.T + ((1 - x) * log_q) @ (1 - y).T) / x.shape[1]
        if self.distance_metric in ("jaccard", "weighted_jaccard"):
            # Binary Jaccard ignores shared absences. Two empty sets have distance 0.
            if not (np.all((left == 0) | (left == 1))
                    and np.all((right == 0) | (right == 1))):
                raise ValueError("Jaccard distances require binary genomes")
            weights = None
            if self.distance_metric == "weighted_jaccard":
                if self.feature_weights is None:
                    raise ValueError("Fit allele frequencies before computing distances")
                weights = self.feature_weights
            if self._use_torch_distances():
                with torch.no_grad():
                    device = self._torch_device()
                    x = self._as_float_tensor(left, device)
                    y = self._as_float_tensor(right, device)
                    w = (torch.ones(x.shape[1], device=device) if weights is None
                         else self._as_float_tensor(weights, device))
                    xw = x * w
                    intersection = xw @ y.T
                    union = xw.sum(1, keepdim=True) + (y @ w).unsqueeze(0) - intersection
                    distances = torch.where(union > 0,
                                            1.0 - intersection / union.clamp_min(1e-30), 0.0)
                    return distances.clamp(0.0, 1.0).cpu().numpy()
            return dist.cdist(left, right, metric="jaccard", **(
                {} if weights is None else {"w": weights}
            ))
        if self.distance_metric == "pca_euclidean":
            if self.pca is None:
                raise ValueError("Fit PCA before computing distances")
            left, right = self.pca.transform(left), self.pca.transform(right)
        if self.distance_metric == "weighted_hamming":
            if self.feature_weights is None:
                raise ValueError("Fit allele frequencies before computing distances")
            if not (np.all((left == 0) | (left == 1))
                    and np.all((right == 0) | (right == 1))):
                raise ValueError("Weighted Hamming requires binary genomes")
            if self._use_torch_distances():
                with torch.no_grad():
                    device = self._torch_device()
                    x = self._as_float_tensor(left, device)
                    y = self._as_float_tensor(right, device)
                    w = self._as_float_tensor(self.feature_weights, device)
                    xw = x * w
                    distances = (xw.sum(1, keepdim=True) + (y @ w).unsqueeze(0)
                                 - 2.0 * (xw @ y.T)) / w.sum()
                    return distances.clamp(0.0, 1.0).cpu().numpy()
            return dist.cdist(left, right, metric="hamming", w=self.feature_weights)
        if self._use_torch_distances():
            with torch.no_grad():
                if self.distance_metric == "hamming":
                    distances = self._normalized_hamming_torch(left, right)
                    return distances.detach().cpu().numpy()
                distances_sq = self._squared_euclidean_torch(left, right)
                if self.distance_metric in ("euclidean", "pca_euclidean"):
                    distances = torch.sqrt(distances_sq)
                else:
                    distances = distances_sq
                return distances.detach().cpu().numpy()

        metric = "euclidean" if self.distance_metric == "pca_euclidean" else self.distance_metric
        return dist.cdist(left, right, metric=metric)

    def _count_inside_d_min(self, left, right):
        if self.distance_metric in ("pca_euclidean", "weighted_hamming", "jaccard", "weighted_jaccard"):
            return np.sum(self._pairwise_distances(left, right) < self.d_min, axis=1)
        if self._use_torch_distances():
            with torch.no_grad():
                if self.distance_metric == "hamming":
                    distances = self._normalized_hamming_torch(left, right)
                    return torch.sum(distances < self.d_min, dim=1).detach().cpu().numpy()
                threshold = self.d_min * self.d_min if self.distance_metric == "euclidean" else self.d_min
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
        Estimate raw membership scores using epsilon or k-nearest distances.
        """
        if self.neighborhood == "epsilon" and not hasattr(self, "d_min"):
            raise ValueError("Call prepare(...) before scoring epsilon signals")

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
            else:
                wrapper = self._get_wrapper(modelWrapper)

        if synthetic_data is not None:
            synthetic_data = self._as_2d(synthetic_data)
            if len(synthetic_data) == 0:
                raise ValueError("synthetic_data must contain at least 1 sample")

        if self.neighborhood == "knn" and self.k_neighbors > n_samples:
            raise ValueError("k_neighbors cannot exceed n_samples")

        counts = np.zeros(len(candidates), dtype=float)
        distance_sums = np.zeros(len(candidates), dtype=float)
        nearest = None
        if self.neighborhood == "knn":
            nearest = np.full((len(candidates), self.k_neighbors), np.inf)
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
                distances = self._pairwise_distances(
                    candidates[start:end], generated_batch
                )
                if self.neighborhood == "epsilon":
                    selected = distances < self.d_min
                    counts[start:end] += np.sum(selected, axis=1)
                    if self.score_function == "log_distance":
                        distance_sums[start:end] += np.sum(
                            np.where(
                                selected,
                                np.log(distances + self.log_delta),
                                0.0,
                            ),
                            axis=1,
                        )
                    elif self.score_function == "average_distance":
                        distance_sums[start:end] += np.sum(
                            np.where(selected, distances, 0.0), axis=1
                        )
                else:
                    combined = np.concatenate(
                        [nearest[start:end], distances], axis=1
                    )
                    nearest[start:end] = np.partition(
                        combined, self.k_neighbors - 1, axis=1
                    )[:, :self.k_neighbors]
                del distances

            del generated_batch
            generated_so_far += generated_count
            self._cleanup_after_batch()

        if self.neighborhood == "knn":
            if self.score_function == "log_distance":
                return -np.sum(np.log(nearest + self.log_delta), axis=1)
            return -np.mean(nearest, axis=1)

        if self.score_function == "count":
            return counts / generated_so_far
        if self.score_function == "log_distance":
            return -distance_sums

        means = np.full(len(candidates), np.inf)
        np.divide(distance_sums, counts, out=means, where=counts > 0)
        return -means

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
