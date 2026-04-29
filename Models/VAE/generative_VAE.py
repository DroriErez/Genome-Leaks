import torch
"""Genome VAE generator adapter.

This module wraps `models_10K_VAE.VAE` with helper methods that load checkpoints
created by `main_VAE.py` and generate synthetic genomes with a single call.

Expected checkpoint format:
    torch.save({
        'epoch': epoch,
        'VAE': vae.state_dict(),
        'Encoder': vae.encoder.state_dict(),
        'Decoder': vae.decoder.state_dict(),
        'data_shape': df_noname.shape[1],
        'latent_size': latent_size,
        'channels': channels,
        'noise_dim': noise_dim,
        'alph': alph,
    }, path)

Usage:
    from Models.VAE.generative_VAE import VAE_generative
    g = VAE_generative('output_dir/VAE_model_last_model')
    samples = g.generate(100)
"""

import torch
import numpy as np
from models.VAE.models_10K_VAE import VAE

try:
    from Gen_Model_Wrapper import GenomeGenerativeModel
except ImportError:
    try:
        from Models.Gen_Model_Wrapper import GenomeGenerativeModel
    except ImportError:
        GenomeGenerativeModel = object


class VAE_generative(GenomeGenerativeModel):
    """Variational Autoencoder wrapper.

    This class provides a minimal API:
    - init(file_path): load pretrained checkpoint
    - generate(n): produce n genomes

    Attributes:
        model: torch.nn.Module VAE model instance or None
        device: torch.device for computation
    """

    def __init__(self, model_path: str = None, device: torch.device = None):
        self.model = None
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        if model_path:
            self.init(model_path)

    def init(self, file_path: str) -> None:
        """Load a VAE model checkpoint.

        Args:
            file_path: checkpoint path produced by main_VAE.py.

        Raises:
            ValueError: if required keys are missing or unsupported checkpoint format.
        """
        checkpoint = torch.load(file_path, weights_only=True, map_location=self.device)

        if isinstance(checkpoint, dict) and 'VAE' in checkpoint and 'Encoder' in checkpoint and 'Decoder' in checkpoint:
            # require metadata to recreate same architecture, or pass manual hyperparameters when using this class
            data_shape = checkpoint.get('data_shape')
            if data_shape is None:
                data_shape = 16383
            latent_size = checkpoint.get('latent_size')
            if latent_size is None:
                latent_size = int((data_shape+1)/(2**12))
            channels = checkpoint.get('channels')
            if channels is None:
                channels = 8
            noise_dim = checkpoint.get('noise_dim')
            if noise_dim is None:                
                noise_dim = 1
            alph = checkpoint.get('alph', 0.01)

            if data_shape is None or latent_size is None or channels is None or noise_dim is None:
                raise ValueError('Checkpoint must include keys data_shape, latent_size, channels, noise_dim')

            self.model = VAE(data_shape=data_shape, latent_size=latent_size, channels=channels, noise_dim=noise_dim, alph=alph)
            self.model.load_state_dict(checkpoint['VAE'])
            self.model.encoder.load_state_dict(checkpoint['Encoder'])
            self.model.decoder.load_state_dict(checkpoint['Decoder'])

        elif isinstance(checkpoint, torch.nn.Module):
            self.model = checkpoint

        else:
            raise ValueError(f'Unsupported checkpoint format: {type(checkpoint)}')

        self.model = self.model.to(self.device)
        self.model.eval()

    def generate(self, n: int) -> np.ndarray:
        """Generate n samples from the VAE.

        The output matches the same post-processing used in training output
        snapshots:
        - clamped at 0
        - rounded to integer (0 or 1)
        - reshaped to (n, sequence_length)

        Args:
            n: number of synthetic genomes to generate.

        Returns:
            ndarray of shape (n, sequence_length) and dtype int.

        Raises:
            ValueError: if model is not loaded.
            AttributeError: if latent_size missing in encoder.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call init() first.")

        latent_size = getattr(self.model.encoder, 'latent_size', None)
        if latent_size is None:
            raise AttributeError('Loaded model encoder contains no latent_size attribute')

        self.model.eval()
        latent_samples = torch.normal(mean=0, std=1, size=(n, 1, latent_size), device = self.device)
        with torch.no_grad():
            generated_genomes = self.model.decoder(latent_samples)
            generated_genomes = generated_genomes.detach().cpu().numpy()
            generated_genomes[generated_genomes < 0] = 0
            generated_genomes = np.rint(generated_genomes)
            generated_genomes = generated_genomes.reshape(generated_genomes.shape[0],generated_genomes.shape[2])

        return generated_genomes