from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple
import numpy as np


class attack(ABC):
    name = "base_attack"

    def __init__(self, threshold: Optional[float] = None) -> None:
        self.threshold = threshold
        self.is_fitted = False

    def fit(self, train_data: np.ndarray, non_train_data: np.ndarray, synthetic_data: np.ndarray):
        self._fit(train_data=train_data, non_train_data=non_train_data, synthetic_data=synthetic_data)
        self.is_fitted = True
        return

    def _fit(self, train_data: np.ndarray, non_train_data: np.ndarray, synthetic_data: np.ndarray) -> None:
        return None

    def is_attack_applicable(self, wrapper_model) -> bool:
        return True

    @abstractmethod
    def score(self, candidates: np.ndarray, **kwargs: Any) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        candidates: np.ndarray,
        threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        scores = self.score(candidates, **kwargs)
        thr = self.threshold if threshold is None else threshold

        if thr is None:
            raise ValueError(
                "No threshold available. Call fit(...) or pass threshold."
            )
        predictions = (scores >= thr)
        return predictions, scores
