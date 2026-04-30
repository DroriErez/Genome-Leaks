from abc import ABC, abstractmethod
import torch
from pathlib import Path
import numpy as np

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class GenomeGenerativeModelWrapper:
    """Wrapper for loading and managing genome generative models.

    This wrapper inspects the model filename prefix and instantiates the
    corresponding generator adapter. Supported prefixes:
      - VAE: `VAE_generative` in `Models/VAE/generative_VAE.py`

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
