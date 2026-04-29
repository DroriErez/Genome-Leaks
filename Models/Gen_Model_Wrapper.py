from abc import ABC, abstractmethod
import torch
from pathlib import Path
import numpy as np

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from models.VAE.generative_VAE import VAE_generative


class GenomeGenerativeModel(ABC):
    """Base class for genome generative models."""
    
    @abstractmethod
    def init(self, file_path: str) -> None:
        """Initialize the model from a file.
        
        Args:
            file_path: Path to the model file
        """
        pass
    
    @abstractmethod
    def generate(self, n: int) -> np.ndarray:
        """Generate n samples from the model.
        
        Args:
            n: Number of samples to generate
            
        Returns:
            Tensor of shape (n, ...) containing generated samples
        """
        pass


class GenomeGenerativeModelWrapper:
    """Wrapper for loading and managing genome generative models.

    This wrapper inspects the model filename prefix and instantiates the
    corresponding generator adapter. Supported prefixes:
      - VAE: `VAE_generative` in `Models/VAE/generative_VAE.py`

    Example:
        wrapper = GenomeGenerativeModelWrapper('output_dir/VAE_model_last_model')
        samples = wrapper.generate(10)
    """
    
    def __init__(self, model_name: str, file_name: str) -> None:
        """Initialize the wrapper and load the model.
        
        Args:
            file_name: Path to the model file
        """
        self.model_name = model_name
        self.file_name = file_name
        self.model_type = None
        self.model = self._load_model()
    
    def _load_model(self) -> GenomeGenerativeModel:
        """Load the appropriate model based on filename prefix."""
        model_type = Path(self.file_name).stem.split('_')[0].upper()
        
        if model_type == "VAE":
            model = VAE_generative()
            self.model_type = "VAE"
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.init(self.file_name)
        return model
    
    def generate(self, n: int) -> torch.Tensor:
        """Generate n samples using the loaded model."""
        return self.model.generate(n)
    
    def get_model_type(self) -> str:
        """Return the type of the loaded model.
        
        Returns:
            str: model type string (e.g., 'VAE').
        """
        return self.model_type
    # Quick smoke test (only run when module executed directly)


if __name__ == "__main__":
    g = GenomeGenerativeModelWrapper("models/saved_models/VAE_model_last_model.pth")
    print(g.generate(5).shape)
