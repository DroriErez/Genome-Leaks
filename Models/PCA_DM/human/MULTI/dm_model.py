
import os
import sys
import math
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd

class GaussianDiffusion:
    """
    Implements the forward diffusion process from DDPM:  
      q(x_t|x_0) = N(α(t) * x_0, σ(t)^2 * I)
    where:
      x_t = α(t) * x_0 + σ(t) * ε,  with ε ~ N(0, I)      
    and the functions α(t) (reflects the total amount of the original signal that remains at each timestep t) 
    and σ(t) (reflects how much noises is added at each timestep t)  are defined via:
      α(t) = sqrt(ᾱ_t) and σ(t) = sqrt(1 - ᾱ_t)
    with:
      ᾱ_t = ∏_{i=1}^{t} (1 - β_i)
    """
    def __init__(
        self,
        num_diffusion_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        device: torch.device = torch.device("cpu")
    ):
        self._num_diffusion_timesteps = num_diffusion_timesteps
        self.device = device
        # Betas
        self._betas = self._get_beta_schedule(
            beta_schedule, beta_start, beta_end, num_diffusion_timesteps
        )
        # Alphas
        alphas_bar = np.cumprod(1.0 - self._betas)
        alphas_bar = np.concatenate(([1.], alphas_bar))  # Makes it convenient for generating sample at t=0 (no noise added)
        # Precompute α(t)=sqrt(ᾱ_t) and σ(t)=sqrt(1 - ᾱ_t) as torch tensors.
        self._alphas = torch.tensor(np.sqrt(alphas_bar), dtype=torch.float32, device=self.device)
        self._sigmas = torch.tensor(np.sqrt(1 - alphas_bar), dtype=torch.float32, device=self.device)

    @staticmethod
    def _get_beta_schedule(beta_schedule: str, beta_start: float, beta_end: float, num_diffusion_timesteps: int) -> np.ndarray:
        """
        Returns a numpy array of beta values given the schedule type.
        Supported schedules: "quad", "linear", "const", "jsd", and "sigmoid".
        """
        def sigmoid(x):
            return 1 / (np.exp(-x) + 1)        
        if beta_schedule == "quad":
            betas = (np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64
            ) ** 2)
        elif beta_schedule == "linear":
            betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
        elif beta_schedule == "const":
            betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
        elif beta_schedule == "jsd":
            betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
        elif beta_schedule == "sigmoid":
            betas = np.linspace(-6, 6, num_diffusion_timesteps)
            betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
        else:
            raise NotImplementedError(f"Unknown beta schedule: {beta_schedule}")
        
        assert betas.shape == (num_diffusion_timesteps,)
        return betas

    @property
    def tmin(self):
        return 1

    @property
    def tmax(self):
        return self._num_diffusion_timesteps

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Returns α(t) = sqrt(ᾱ_t) for the given timesteps."""
        return self._alphas[t.long()]

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Returns σ(t) = sqrt(1 - ᾱ_t) for the given timesteps."""
        return self._sigmas[t.long()]

    def sample(self, x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """ Draws samples from the forward diffusion process q(x_t|x_0):
          x_t = α(t) * x0 + σ(t) * ε,
        where ε ~ N(0, I).
        """
        alpha_t = self.alpha(t).view(-1, 1) 
        sigma_t = self.sigma(t).view(-1, 1)
        return alpha_t * x0 + sigma_t * eps

class TimeSampler:
    def __init__(self, t_min: int, t_max: int):
        """
        Initializes the TimeSampler with the minimum and maximum timestep.
        Args:
            t_min (int): The minimum timestep (inclusive).
            t_max (int): The maximum timestep (exclusive).
        """
        self.t_min = t_min
        self.t_max = t_max

    def sample(self, size: int, strategy: str = "antithetic") -> torch.Tensor:
        """
        Samples timesteps according to the specified strategy.
        Args:
            size (int): The number of timesteps to generate.
            strategy (str): Sampling strategy; either "antithetic" (default) or "uniform".
        Returns:
            torch.Tensor: A tensor of sampled timesteps.
        """
        if strategy == "uniform":
            # Simply sample uniformly between t_min and t_max.
            return torch.randint(low=self.t_min, high=self.t_max, size=(size,))
        elif strategy == "antithetic":
            # Calculate the number of samples needed from one side (round up if odd)
            half_n = size // 2 + (size % 2)
            # Sample half_n timesteps uniformly.
            t_half = torch.randint(low=self.t_min, high=self.t_max, size=(half_n,))
            # Compute antithetic counterparts.
            t_antithetic = self.t_max - t_half - 1
            # Concatenate both and slice to exactly 'size' samples.
            t_full = torch.cat([t_half, t_antithetic], dim=0)[:size]
            return t_full
        else:
            raise ValueError(f"Unknown strategy '{strategy}'. Use 'uniform' or 'antithetic'.")

class NoisePredictor(nn.Module):
    def __init__(self, data_dim, time_emb_dim,
                 hidden_dim_1, hidden_dim_2, hidden_dim_3,
                 total_timesteps: int = 1000, device=None):
        """
        Args:
            data_dim (int): Dimension of the data (e.g. 4819).
            time_emb_dim (int): Dimension of the time embedding. 
                                If 1, a normalized timestep is used; otherwise, a sinusoidal embedding.
            hidden_dim_1 (int): Number of hidden units in the first layer.
            hidden_dim_2 (int): Number of hidden units in the second layer.
            hidden_dim_3 (int): Number of hidden units in the third layer.
            total_timesteps (int): Total number of diffusion timesteps.
            device: Optional device on which to allocate the network parameters.
        """
        super(NoisePredictor, self).__init__()
        self.data_dim = data_dim
        self.time_emb_dim = time_emb_dim
        self.total_timesteps = total_timesteps
        
        # The overall input dimension is noisy PCA data plus the time embedding.
        self.input_dim = data_dim + time_emb_dim
        
        # First layer: maps input_dim to hidden_dim_1.
        self.fc1 = nn.Linear(self.input_dim, hidden_dim_1, device=device)
        self.norm1 = nn.LayerNorm(hidden_dim_1, device=device)
        
        # Second layer: inject the time embedding again.
        self.fc2 = nn.Linear(
            hidden_dim_1 + time_emb_dim, hidden_dim_2, device=device
        )
        self.norm2 = nn.LayerNorm(hidden_dim_2, device=device)
        
        # Third layer: maps from hidden_dim_2 to hidden_dim_3.
        self.fc3 = nn.Linear(hidden_dim_2, hidden_dim_3, device=device)
        self.norm3 = nn.LayerNorm(hidden_dim_3, device=device)
        
        # Output layer: maps from hidden_dim_3 to data_dim (the predicted noise).
        self.out = nn.Linear(hidden_dim_3, data_dim, device=device)
        
        self.activation = nn.ReLU()
        
        # Residual connection: project the initial input to data_dim.
        self.res_proj = nn.Linear(self.input_dim, data_dim, device=device)

    def get_timestep_embedding(self, timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
        """
        Generate a time embedding for each timestep.
        If embedding_dim == 1, returns a normalized timestep (t / total_timesteps).
        Otherwise, returns a sinusoidal embedding as in the DDPM paper.
        
        Args:
            timesteps (torch.Tensor): 1D tensor of timesteps (shape: [batch_size]).
            embedding_dim (int): The desired dimension of the embedding.
            
        Returns:
            torch.Tensor: A tensor of shape [batch_size, embedding_dim].
        """
        assert timesteps.ndim == 1, "timesteps should be a 1D tensor"
        if embedding_dim == 1:
            return timesteps.float()[:, None] / self.total_timesteps
        else:
            half_dim = embedding_dim // 2
            emb_factor = math.log(10000) / (half_dim - 1)
            emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb_factor)
            emb = timesteps.float()[:, None] * emb[None, :]
            emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
            if embedding_dim % 2 == 1:
                emb = F.pad(emb, (0, 1))
            return emb

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Noisy data of shape [batch_size, data_dim].
            t (torch.Tensor): Timestep tensor of shape [batch_size].
            
        Returns:
            torch.Tensor: Predicted noise of shape [batch_size, data_dim].
        """
        # Compute time embedding.
        t_emb = self.get_timestep_embedding(t, self.time_emb_dim)  # [batch_size, time_emb_dim]
        
        # Concatenate the noisy data and time embedding.
        x_in = torch.cat([x, t_emb], dim=1)  # [batch_size, input_dim]
        
        # First layer.
        h1 = self.fc1(x_in)
        h1 = self.norm1(h1)
        h1 = self.activation(h1)
        
        # Inject the time embedding again into the intermediate layer.
        h1_inj = torch.cat([h1, t_emb], dim=1)
        h2 = self.fc2(h1_inj)
        h2 = self.norm2(h2)
        h2 = self.activation(h2)
        
        # Third layer.
        h3 = self.fc3(h2)
        h3 = self.norm3(h3)
        h3 = self.activation(h3)
        
        # Output layer.
        out = self.out(h3)  # [batch_size, data_dim]
        
        # Residual connection: project the original concatenated input to data_dim.
        res = self.res_proj(x_in)
        
        return out + res

def bcast_right(x: torch.Tensor, ndim: int) -> torch.Tensor:
    """Util function for broadcasting to the right."""
    if x.ndim > ndim:
        raise ValueError(f'Cannot broadcast a value with {x.ndim} dims to {ndim} dims.')
    elif x.ndim < ndim:
        difference = ndim - x.ndim
        return x.view(x.shape + (1,) * difference)
    else:
        return x

class DDPM(nn.Module):
    """The forward diffusion and backward denoising process for DDPM."""
    def __init__(self, snp_dim, diffusion_process, time_sampler, noise_predictor):
        super(DDPM, self).__init__()
        self._snp_dim = snp_dim
        self._diffusion = diffusion_process
        self._time_sampler = time_sampler
        self._noise_predictor = noise_predictor

    def loss(self, x0: torch.Tensor) -> torch.Tensor:
        """
        Computes the MSE loss between the true noise and the noise predicted by the model.
        
        Args:
            x0 (torch.Tensor): Clean data of shape [batch_size, snp_dim].
            
        Returns:
            torch.Tensor: The mean squared error loss.
        """
        t = self._time_sampler.sample(size=x0.shape[0]).to(x0.device)  # Sample timesteps
        eps = torch.randn_like(x0, device=x0.device)                      # Sample noise
        xt = self._diffusion.sample(x0, t, eps).to(x0.device)             # Generate noisy data
        predicted_noise = self._noise_predictor(xt, t)                      # Predict noise
        loss = torch.mean((predicted_noise - eps) ** 2)
        return loss

    def one_reverse_step(self, xt: torch.Tensor, t: int) -> torch.Tensor:
        """
        Computes one reverse diffusion step to denoise xt.
        
        This function computes a sample from the Gaussian distribution
        p(x_{t-1} | x_t, x0_pred) based on the current noisy sample xt, the current timestep t,
        
        Args:
            xt (torch.Tensor): The current noisy input of shape [batch_size, snp_dim].
            t (int): The current timestep (should be >= 1).
        
        Returns:
            torch.Tensor: A denoised sample from p(x_{t-1}| x_t, x0_pred).
        """
        # Create a tensor of timesteps with shape [batch_size]
        t_tensor = t * torch.ones((xt.shape[0],), dtype=torch.int32, device=xt.device)
        # Predict the noise.
        eps_pred = self._noise_predictor(xt, t_tensor)
        # Compute the scaling factor for reversing the diffusion.
        sqrt_a_t = self._diffusion.alpha(t_tensor) / self._diffusion.alpha(t_tensor - 1)
        inv_sqrt_a_t = bcast_right(1.0 / sqrt_a_t, xt.ndim)
        # Compute the variance term for the reverse process.
        beta_t = 1.0 - sqrt_a_t ** 2
        beta_t = bcast_right(beta_t, xt.ndim)
        inv_sigma_t = bcast_right(1.0 / self._diffusion.sigma(t_tensor), xt.ndim)
        # Compute the mean of the reverse Gaussian.
        mean = inv_sqrt_a_t * (xt - beta_t * inv_sigma_t * eps_pred)
        # Standard deviation is the square root of beta_t.
        std = torch.sqrt(beta_t)
        z = torch.randn_like(xt)
        return mean + std * z

    def sample(self, sample_size: int, device: torch.device) -> torch.Tensor:
        """
        Generates synthetic samples by running the reverse diffusion process from pure noise.
        
        Args:
            sample_size (int): The number of samples to generate.
        
        Returns:
            torch.Tensor: Generated samples of shape [sample_size, snp_dim].
        """
        with torch.no_grad():
            x = torch.randn(sample_size, self._snp_dim, device=device)
            # Iterate from the maximum timestep down to 1.
            for t in range(self._diffusion.tmax, 0, -1):
                x = self.one_reverse_step(x, t)
        return x
