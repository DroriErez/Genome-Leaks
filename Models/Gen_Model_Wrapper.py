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
        """Generate fresh samples using the loaded model."""
        return self.model.generate(n)

    @staticmethod
    def _safe_cache_name(value) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "model"

    def _default_synthetic_cache_key(self) -> str:
        model_path = Path(self.file_name)
        if model_path.is_file():
            return f"{model_path.stem}_{model_path.stat().st_mtime_ns}"
        return self.model_name

    def get_synthetic_data(
        self,
        n: int,
        cache_dir=None,
        use_cache: bool = True,
        cache_key=None,
        batch_size: int = 500,
        expected_feature_shape=None,
    ) -> np.ndarray:
        """Return cached synthetic samples, extending or creating the cache.

        ``generate`` always produces fresh samples. This method is the explicit
        reusable-dataset API shared by quality evaluation and attacks.
        """
        if n < 1:
            raise ValueError("n must be at least 1")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        self.synthetic_cache_path = None
        if not use_cache or cache_dir is None:
            print(f"Synthetic cache disabled. Generating with model: count={n}")
            return np.asarray(self.generate(n=n))

        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_name = self._safe_cache_name(
            cache_key or self._default_synthetic_cache_key()
        )
        cache_path = cache_dir / f"{cache_name}_synthetic_{n}.npy"
        pattern = re.compile(rf"{re.escape(cache_name)}_synthetic_(\d+)\.npy")
        larger = []
        smaller = []
        for candidate in cache_dir.glob(f"{cache_name}_synthetic_*.npy"):
            match = pattern.fullmatch(candidate.name)
            if not match:
                continue
            count = int(match.group(1))
            (larger if count >= n else smaller).append((count, candidate))

        def load_valid(candidate):
            try:
                cached = np.load(candidate, mmap_mode="r")
            except (OSError, ValueError) as error:
                print(f"Ignoring unreadable synthetic cache {candidate}: {error}")
                return None
            shape_matches = (
                expected_feature_shape is None
                or tuple(cached.shape[1:]) == tuple(expected_feature_shape)
            )
            if cached.ndim == 2 and len(cached) > 0 and shape_matches:
                return cached
            print(
                f"Ignoring synthetic cache {candidate} with shape {cached.shape}; "
                f"expected feature shape {expected_feature_shape}."
            )
            return None

        for _, candidate in sorted(larger):
            cached = load_valid(candidate)
            if cached is not None and len(cached) >= n:
                self.synthetic_cache_path = candidate
                print(
                    "Using cached synthetic dataset: "
                    f"requested={n}, available={len(cached)}, path={candidate}"
                )
                return cached[:n]

        seed = None
        for _, candidate in sorted(smaller, reverse=True):
            seed = load_valid(candidate)
            if seed is not None:
                print(
                    f"Using cached synthetic dataset as generation seed: "
                    f"available={len(seed)}, requested={n}, path={candidate}"
                )
                break

        temp_path = cache_path.with_suffix(".tmp.npy")
        if temp_path.exists():
            temp_path.unlink()

        if seed is None:
            print(
                "No compatible synthetic cache found. Generating synthetic "
                f"dataset with model: count={n}"
            )
            first = np.asarray(self.generate(n=min(batch_size, n)))
        else:
            first = np.asarray(seed[:n])
            if len(first) < n:
                print(
                    "Generating missing synthetic samples with model: "
                    f"count={n - len(first)}"
                )
        if first.ndim != 2 or len(first) == 0:
            raise ValueError(f"Generated synthetic data has invalid shape {first.shape}")
        if expected_feature_shape is not None and tuple(first.shape[1:]) != tuple(expected_feature_shape):
            raise ValueError(
                f"Synthetic samples have feature shape {first.shape[1:]}; "
                f"expected {expected_feature_shape}"
            )

        output = np.lib.format.open_memmap(
            temp_path, mode="w+", dtype=first.dtype, shape=(n, *first.shape[1:])
        )
        output[:len(first)] = first
        written = len(first)
        print(f"Synthetic cache progress: {written}/{n}")
        while written < n:
            current = min(batch_size, n - written)
            generated = np.asarray(self.generate(n=current))[:current]
            if generated.ndim != 2 or generated.shape[1:] != first.shape[1:]:
                raise ValueError(
                    f"Generated batch shape {generated.shape} does not match "
                    f"feature shape {first.shape[1:]}"
                )
            output[written:written + len(generated)] = generated
            written += len(generated)
            print(f"Synthetic cache progress: {written}/{n}")
        output.flush()
        del output
        temp_path.replace(cache_path)
        self.synthetic_cache_path = cache_path
        print(f"Saved synthetic dataset cache: {cache_path}")
        return np.load(cache_path, mmap_mode="r")
    
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
        for model_type in ("PCA_DM", "WGAN", "VAE", "AC_GAN"):
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
