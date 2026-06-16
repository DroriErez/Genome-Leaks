"""Genome WGAN generator adapter.

This module wraps `models_10K.ConvGenerator` with helper methods that load
checkpoints created by `main_WGAN.py` and generate synthetic genomes with a
single call.

Expected checkpoint format:
    torch.save({
        'epoch': epoch,
        'Generator': netG.state_dict(),
        'Critic': netC.state_dict(),
        'G_optimizer': g_optimizer.state_dict(),
        'C_optimizer': c_optimizer.state_dict(),
    }, path)
"""

import gc

import torch
import numpy as np

from models.Gen_Model_Wrapper import GenomeGenerativeModelWrapper
from models.WGAN.models_10K import ConvDiscriminator, ConvGenerator


class WGAN_generative(GenomeGenerativeModelWrapper):
    """Wasserstein GAN wrapper for synthetic genome generation."""

    def __init__(
        self,
        model_path: str = None,
        device: torch.device = None,
        generation_batch_size: int = 16,
    ):
        super().__init__(model_path)
        self.model = None
        self.critic = None
        self.model_architecture = "WGAN"
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.generation_batch_size = generation_batch_size
        self.data_shape = None
        self.latent_size = None
        self.channels = None
        self.noise_dim = None
        self.alph = None
        self.gpu = None
        self.pack_m = None

        if model_path:
            self.init(model_path)

    def init(self, file_path: str) -> None:
        """Load a WGAN generator and critic checkpoint."""
        checkpoint = self._load_checkpoint(file_path)

        if isinstance(checkpoint, dict) and "Generator" in checkpoint:

            self.alph = 0.01 #alpha value for LeakyReLU

            self.channels = 10 #channel multiplier which dictates the number of channels for all layers
            self.gpu = 1 #number of GPUs
            self.noise_dim = 2 #dimension of noise for each noise vector
            latent_depth_factor = 12 #14 for 65535 SNP data and 12 for 16383 zero padded SNP data
            self.data_shape = 16383 #set the data shape
            self.latent_size = int((self.data_shape+1)/(2**latent_depth_factor)) #set the latent_size
            self.pack_m = 3 #number of samples to pack together for the critic


            self.model = ConvGenerator(latent_size=self.latent_size, data_shape=self.data_shape, gpu=self.gpu, device=self.device, channels=self.channels, noise_dim=self.noise_dim, alph=self.alph)
            self.model = self.model.float()
            if (self.device.type == 'cuda') and (self.gpu > 1):
                self.model = torch.nn.DataParallel(self.model, list(range(self.gpu)))
            self.model.to(self.device)
            self.model.load_state_dict(checkpoint['Generator'])

            # self.data_shape = checkpoint.get("data_shape", 16383)
            # latent_depth_factor = checkpoint.get("latent_depth_factor", 12)
            # self.latent_size = checkpoint.get("latent_size", int((self.data_shape + 1) / (2**latent_depth_factor)))
            # self.channels = checkpoint.get("channels", 10)
            # self.noise_dim = checkpoint.get("noise_dim", 2)
            # self.alph = checkpoint.get("alph", 0.01)
            # self.gpu = checkpoint.get("gpu", 1)
            # self.pack_m = checkpoint.get("pack_m", 3)

            # self.model = ConvGenerator(
            #     latent_size=self.latent_size,
            #     data_shape=self.data_shape,
            #     gpu=self.gpu,
            #     device=self.device,
            #     channels=self.channels,
            #     noise_dim=self.noise_dim,
            #     alph=self.alph,
            # )

            # generator_state = self._clean_state_dict(checkpoint["Generator"])
            # self.model.load_state_dict(generator_state)

            ## Create the critic
            self.critic = ConvDiscriminator(data_shape=self.data_shape, latent_size=self.latent_size, gpu=self.gpu, pack_m = self.pack_m, device=self.device, channels=self.channels, alph=self.alph).to(self.device)
            self.critic = self.critic.float()
            if (self.device.type == 'cuda') and (self.gpu > 1):
                self.critic = torch.nn.DataParallel(self.critic, list(range(self.gpu)))
            self.critic.to(self.device)
            # if "Critic" in checkpoint and checkpoint["Critic"]:
            #     self.critic = ConvDiscriminator(
            #         data_shape=self.data_shape,
            #         latent_size=self.latent_size,
            #         gpu=self.gpu,
            #         device=self.device,
            #         pack_m=self.pack_m,
            #         channels=self.channels,
            #         alph=self.alph,
            #     )
            #     critic_state = self._clean_state_dict(checkpoint["Critic"])
            self.critic.load_state_dict(checkpoint["Critic"])

        elif isinstance(checkpoint, torch.nn.Module):
            self.model = checkpoint

        else:
            raise ValueError(f"Unsupported checkpoint format: {type(checkpoint)}")

        self.model = self.model.to(self.device)
        self.model.eval()
        if self.critic is not None:
            self.critic = self.critic.to(self.device)
            self.critic.eval()

        del checkpoint
        gc.collect()

    def _load_checkpoint(self, file_path: str):
        """Load checkpoint with memory mapping when available."""
        try:
            return torch.load(
                file_path,
                weights_only=True,
                map_location=self.device,
                mmap=True,
            )
        except TypeError:
            return torch.load(
                file_path,
                weights_only=True,
                map_location=self.device,
            )

    @staticmethod
    def _clean_state_dict(state_dict: dict) -> dict:
        """Remove DataParallel prefixes from checkpoint keys if present."""
        if any(key.startswith("module.") for key in state_dict):
            return {
                key.removeprefix("module."): value
                for key, value in state_dict.items()
            }
        return state_dict

    def noise_generator(self, size: int, noise_count: int = 6) -> list[torch.Tensor]:
        """Create the per-scale noise tensors expected by ConvGenerator."""
        noise_list = []
        for i in range(2, noise_count * 2 + 1, 2):
            noise = torch.normal(
                mean=0,
                std=1,
                size=(size, self.model.noise_dim, self.model.latent_size * (2**i) - 1),
                device=self.device,
            )
            noise_list.append(noise)
        return noise_list

    def get_critic(self) -> torch.nn.Module:
        """Return the loaded WGAN critic."""
        if self.critic is None:
            raise ValueError("Critic not loaded. Check that the checkpoint contains a non-empty 'Critic' state dict.")
        return self.critic

    def critic_score(self, samples) -> np.ndarray:
        """Score packed samples with the WGAN critic.

        Args:
            samples: Tensor or ndarray with shape (batch, pack_m, data_shape).
                Shape (batch, data_shape) is accepted only when pack_m == 1.

        Returns:
            ndarray of critic scores with shape (batch,).
        """
        critic = self.get_critic()
        samples = torch.as_tensor(samples, dtype=torch.float32, device=self.device)

        if samples.ndim == 2:
            if self.pack_m != 1:
                raise ValueError(
                    f"Critic expects packed samples with shape (batch, {self.pack_m}, {self.data_shape}); "
                    "2D samples are only valid when pack_m == 1."
                )
            samples = samples[:, np.newaxis, :]
        elif samples.ndim != 3:
            raise ValueError("Critic samples must have shape (batch, pack_m, data_shape)")

        if samples.shape[1] != self.pack_m or samples.shape[2] != self.data_shape:
            raise ValueError(
                f"Critic expects shape (batch, {self.pack_m}, {self.data_shape}), got {tuple(samples.shape)}"
            )

        critic.eval()
        with torch.no_grad():
            scores = critic(samples).detach().cpu().numpy()

        return scores.reshape(scores.shape[0])

    def _generate_batch(self, n: int) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not loaded. Call init() first.")

        latent_samples = torch.normal(
            mean=0,
            std=1,
            size=(n, self.model.noise_dim, self.model.latent_size),
            device=self.device,
        )
        noise_list = self.noise_generator(n)

        with torch.no_grad():
            generated_genomes = self.model(latent_samples, noise_list)
            generated_genomes = generated_genomes.detach().cpu().numpy()
            generated_genomes[generated_genomes < 0] = 0
            generated_genomes = np.rint(generated_genomes)
            generated_genomes = generated_genomes.reshape(
                generated_genomes.shape[0],
                generated_genomes.shape[2],
            )

        return generated_genomes.astype(np.int8, copy=False)

    def generate(self, n: int) -> np.ndarray:
        """Generate n synthetic genomes from the WGAN generator."""
        if self.model is None:
            raise ValueError("Model not loaded. Call init() first.")

        self.model.eval()
        generated_batches = []
        generated_so_far = 0

        while generated_so_far < n:
            current_batch_size = min(
                self.generation_batch_size,
                n - generated_so_far,
            )
            generated_batches.append(self._generate_batch(current_batch_size))
            generated_so_far += current_batch_size

        return np.vstack(generated_batches)
