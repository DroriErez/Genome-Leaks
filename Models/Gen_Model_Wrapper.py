from abc import ABC, abstractmethod
import torch
from pathlib import Path
import numpy as np
import pandas as pd
import re

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class GenomeGenerativeModelWrapper:
    """Wrapper for loading and managing genome generative models.

    This wrapper inspects the model filename prefix and instantiates the
    corresponding generator adapter. Supported prefixes:
      - VAE: `VAE_generative` in `Models/VAE/generative_VAE.py`
      - WGAN: `WGAN_generative` in `Models/WGAN/generative_WGAN.py`

    Example:
        wrapper = GenomeGenerativeModelWrapper('output_dir/VAE_model_last_model')
        samples = wrapper.generate(10)
    """
    
    def __init__(self, file_name: str) -> None:
        """Initialize the wrapper and load the model.
        
        Args:
            file_name: Path to the model file
        """
        self.model_name = Path(file_name).stem
        self.file_name = file_name
        self.model_architecture = None
        self.model = None
    
    def generate(self, n: int) -> torch.Tensor:
        """Generate n samples using the loaded model."""
        return self.model.generate(n)
    
    def get_model_architecture(self) -> str:
        """Return the architecture of the loaded model.

        Returns:
            str: model architecture string (e.g., 'VAE').
        """
        return self.model_architecture

    @staticmethod
    def infer_model_type(model_name) -> str:
        """Infer the high-level generator type from a checkpoint/model name."""
        normalized_name = str(model_name).upper().replace("-", "_")
        for model_type in ("WGAN", "VAE", "AC_GAN"):
            if model_type in normalized_name:
                return model_type
        if "GAN" in normalized_name:
            return "GAN"
        return "unknown"

    @staticmethod
    def infer_model_epochs(model_name) -> str:
        """Infer epochs from names like WGAN_model_1000 or return 'last'."""
        model_name = str(model_name)
        if "last_model" in model_name.lower():
            return "last"
        match = re.search(r"_model_(\d+)", model_name)
        return match.group(1) if match else "unknown"

    @staticmethod
    def format_model_title(model_name) -> str:
        """Return a compact plot-title suffix such as 'WGAN, 1000 epochs'."""
        model_type = GenomeGenerativeModelWrapper.infer_model_type(model_name)
        epochs = GenomeGenerativeModelWrapper.infer_model_epochs(model_name)
        if epochs == "unknown":
            return model_type
        if epochs == "last":
            return f"{model_type}, last"
        return f"{model_type}, {epochs} epochs"

    def get_model_type(self) -> str:
        return self.infer_model_type(self.model_name)

    def get_model_epochs(self) -> str:
        return self.infer_model_epochs(self.model_name)

    def get_model_title_suffix(self) -> str:
        return self.format_model_title(self.model_name)

    def get_attack_dataset_paths(self, models_folder: str, base: str) -> dict:
        """Return train/eval/test dataset paths used by attack runners."""
        models_path = Path(models_folder)
        return {
            "train": models_path / f"{base}_train.hapt",
            "eval": models_path / f"{base}_eval.hapt",
            "test": models_path / f"{base}_test.hapt",
        }

    def load_attack_dataset(self, path, nrows=None) -> np.ndarray:
        """Load a default .hapt-style attack dataset as a numeric matrix."""
        df = pd.read_csv(path, sep=" ", header=None, nrows=nrows)
        values = df.drop(df.columns[0:2], axis=1).values
        return self._as_attack_matrix(values, path)

    def load_attack_labels(self, path, nrows=None):
        """Load optional class labels for label-aware attacks.

        Default .hapt-style datasets do not expose class labels in a model-
        agnostic way, so wrappers can override this when labels are available.
        """
        return None

    @staticmethod
    def _as_attack_matrix(values, source) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(
                f"Attack dataset must be 2D, got shape {matrix.shape} from {source}"
            )
        if matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError(
                f"Attack dataset must be non-empty, got shape {matrix.shape} from {source}"
            )
        if not np.isfinite(matrix).all():
            raise ValueError(f"Attack dataset contains NaN or infinite values: {source}")
        return matrix
