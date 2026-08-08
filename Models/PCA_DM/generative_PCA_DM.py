"""Adapter for PCA-space diffusion-model checkpoints."""

from pathlib import Path
import re

import numpy as np
import torch

from models.Gen_Model_Wrapper import GenomeGenerativeModelWrapper
from models.PCA_DM.human.MULTI.dm_model import (
    DDPM,
    GaussianDiffusion,
    NoisePredictor,
    TimeSampler,
)
from utils.device import get_device


class PCA_DM_generative(GenomeGenerativeModelWrapper):
    """Load a PCA-DM checkpoint and expose generation and attack losses."""

    def __init__(self, model_path=None, device=None, generation_batch_size=128):
        super().__init__(model_path)
        self.model = None
        self.model_architecture = "PCA_DM"
        self.device = device or get_device()
        if generation_batch_size < 1:
            raise ValueError("generation_batch_size must be at least 1")
        self.generation_batch_size = int(generation_batch_size)
        if model_path:
            self.init(model_path)

    @staticmethod
    def _pca_path(model_path):
        path = Path(model_path)
        match = re.fullmatch(r"PCA_DM_model_(\d+)\.pth", path.name)
        if not match:
            raise ValueError(
                "PCA-DM checkpoints must be named PCA_DM_model_<epoch>.pth"
            )
        return path.with_name(f"PCA_DM_{match.group(1)}_pca.npz")

    def init(self, file_path):
        checkpoint = torch.load(
            file_path, map_location=self.device, weights_only=False
        )
        required = {"model_state_dict", "latent_dim", "snp_dim", "config"}
        missing = required.difference(checkpoint)
        if missing:
            raise ValueError(f"PCA-DM checkpoint is missing keys: {sorted(missing)}")

        config = checkpoint["config"]
        latent_dim = int(checkpoint["latent_dim"])
        diffusion = GaussianDiffusion(
            num_diffusion_timesteps=int(config["num_timesteps"]),
            device=self.device,
        )
        time_sampler = TimeSampler(diffusion.tmin, diffusion.tmax)
        predictor = NoisePredictor(
            latent_dim,
            int(config["time_embedding_dim"]),
            int(config["hidden_dim_1"]),
            int(config["hidden_dim_2"]),
            int(config["hidden_dim_3"]),
            int(config["num_timesteps"]),
            device=self.device,
        )
        self.model = DDPM(latent_dim, diffusion, time_sampler, predictor)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device).eval()

        pca_path = self._pca_path(file_path)
        if not pca_path.is_file():
            raise FileNotFoundError(
                f"PCA transform required by {Path(file_path).name} was not found: "
                f"{pca_path}"
            )
        with np.load(pca_path) as pca:
            self.pca_components = pca["components"].astype(np.float32)
            self.pca_mean = pca["mean"].astype(np.float32)
        if self.pca_components.shape != (latent_dim, int(checkpoint["snp_dim"])):
            raise ValueError(
                "PCA artifact dimensions do not match the PCA-DM checkpoint: "
                f"{self.pca_components.shape}"
            )

    def _to_pca_scores(self, genomes):
        genomes = np.asarray(genomes, dtype=np.float32)
        if genomes.ndim != 2 or genomes.shape[1] != self.pca_mean.size:
            raise ValueError(
                f"Expected genomes shaped (n, {self.pca_mean.size}), got {genomes.shape}"
            )
        return (genomes - self.pca_mean) @ self.pca_components.T

    def diffusion_losses(
        self,
        genomes,
        n_repeats=8,
        batch_size=32,
        timestep=None,
    ):
        """Return Monte Carlo estimates of DDPM training loss per genome."""
        if self.model is None:
            raise ValueError("Model not loaded. Call init() first.")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        scores = self._to_pca_scores(genomes)
        if len(scores) == 0:
            raise ValueError("genomes must contain at least one sample")
        losses = []
        diffusion = self.model._diffusion
        predictor = self.model._noise_predictor
        time_sampler = self.model._time_sampler
        if timestep is not None:
            timestep = int(timestep)
            if not 0 <= timestep <= diffusion.tmax:
                raise ValueError(
                    f"timestep must be between 0 and {diffusion.tmax}, got {timestep}"
                )
        repeats = max(1, int(n_repeats))
        with torch.no_grad():
            for start in range(0, len(scores), batch_size):
                x0 = torch.from_numpy(scores[start:start + batch_size]).to(self.device)
                batch_losses = torch.zeros(x0.shape[0], device=self.device)
                for _ in range(repeats):
                    # Match DDPM.loss sampling unless a fixed attack timestep is set.
                    if timestep is None:
                        t = time_sampler.sample(x0.shape[0]).to(self.device)
                    else:
                        t = torch.full(
                            (x0.shape[0],), timestep, device=self.device,
                            dtype=torch.long,
                        )
                    eps = torch.randn_like(x0)
                    xt = diffusion.sample(x0, t, eps)
                    predicted = predictor(xt, t)
                    batch_losses += (predicted - eps).square().mean(dim=1)
                losses.append((batch_losses / repeats).cpu().numpy())
        return np.concatenate(losses)

    def generate(self, n):
        if self.model is None:
            raise ValueError("Model not loaded. Call init() first.")
        if n < 1:
            raise ValueError("n must be at least 1")
        generated = []
        with torch.no_grad():
            for start in range(0, n, self.generation_batch_size):
                size = min(self.generation_batch_size, n - start)
                scores = self.model.sample(size, self.device).cpu().numpy()
                genomes = scores @ self.pca_components + self.pca_mean
                generated.append((np.clip(genomes, 0, 1) >= 0.5).astype(np.int8))
        return np.concatenate(generated)
