"""
Monte Carlo Membership Inference Attack

This attack uses Monte Carlo sampling to estimate whether a candidate 
is a member of the training set. It samples from the model and compares
the reconstruction loss/distance of candidates against sampled data.
"""

from typing import Any, Optional, Tuple
import numpy as np
from scipy.spatial import distance as dist
from scipy.stats import percentileofscore
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
        threshold: Optional[float] = None,
    ) -> None:
        """
        Initialize Monte Carlo Attack.

        Args:
            n_samples: Number of Monte Carlo samples to draw
            distance_metric: Distance metric to use ('euclidean', 'cosine', etc.)
            threshold: Decision threshold for membership (higher = member)
        """
        super().__init__(threshold=threshold)
        self.n_samples = n_samples
        self.distance_metric = distance_metric
        self.ref_distances = None
        self.mean_distance = None
        self.std_distance = None
        self.non_train_distances = None
        self.synthetic_distances = None
        self.combined_distances = None
        self.synthetic_mean_distance = None
        self.synthetic_std_distance = None

    def fit(self, train_data: np.ndarray, non_train_data: np.ndarray, synthetic_data: np.ndarray):
        """Override fit to accept train_data, non_train_data, and synthetic_data."""
        self._fit(train_data=train_data, non_train_data=non_train_data, synthetic_data=synthetic_data)
        self.is_fitted = True
        return

    def _fit(self, train_data, non_train_data, synthetic_data):
        if len(synthetic_data) < 2:
            raise ValueError("Synthetic data must contain at least 2 samples")

        distances_tr = []
        distances_non = []

        for x in train_data:
            d = dist.cdist([x], synthetic_data, metric=self.distance_metric)[0]
            distances_tr.append(np.min(d))

        for x in non_train_data:
            d = dist.cdist([x], synthetic_data, metric=self.distance_metric)[0]
            distances_non.append(np.min(d))

        distances = np.concatenate([distances_tr, distances_non])
        self.d_min = np.median(distances)
        
    def score(self, candidates, synthetic_data=None, **kwargs):
        if synthetic_data is None:
            raise ValueError("synthetic_data must be provided")

        scores = []

        for x in candidates:
            dists = dist.cdist([x], synthetic_data, metric=self.distance_metric)[0]
            score = np.mean(dists < self.d_min)   # MC eps score
            scores.append(score)

        return np.array(scores)
    
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
        thr = self.threshold if threshold is None else threshold

        if thr is None:
            raise ValueError(
                "No threshold available. Call fit(...) or pass threshold."
            )

        predictions = scores >= thr
        return predictions, scores
