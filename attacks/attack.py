from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple
import numpy as np


class attack(ABC):
    name = "base_attack"

    def __init__(self) -> None:
        self.is_fitted = False

    def fit(self, non_train_data: np.ndarray, thr: float = 0.5, modelWrapper=None) -> None:
        self.modelWrapper = modelWrapper
        self.threshold = thr
        self._fit(non_train_data=non_train_data, modelWrapper=modelWrapper)
        self.is_fitted = True
        return

    def _fit(self, non_train_data: np.ndarray, modelWrapper=None) -> None:
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
