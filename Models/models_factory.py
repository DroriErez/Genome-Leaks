
from pathlib import Path

from models.Gen_Model_Wrapper import GenomeGenerativeModelWrapper
from models.AC_GAN.generative_ACGAN import ACGAN_generative
from models.VAE.generative_VAE import VAE_generative
from models.WGAN.generative_WGAN import WGAN_generative


def create_model_wrapper(
    file_name: str,
    device=None,
    generation_batch_size=None,
) -> 'GenomeGenerativeModelWrapper':
    model_stem = Path(file_name).stem
    model_type = model_stem.split('_')[0].upper()
    
    if model_stem.startswith("AC_GAN"):
        kwargs = {"model_path": file_name}
        if generation_batch_size is not None:
            kwargs["generation_batch_size"] = generation_batch_size
        return ACGAN_generative(**kwargs)
    elif model_type == "VAE":
        kwargs = {"model_path": file_name, "device": device}
        if generation_batch_size is not None:
            kwargs["generation_batch_size"] = generation_batch_size
        return VAE_generative(**kwargs)
    elif model_type == "WGAN":
        kwargs = {"model_path": file_name, "device": device}
        if generation_batch_size is not None:
            kwargs["generation_batch_size"] = generation_batch_size
        return WGAN_generative(**kwargs)
    else:
        raise ValueError(f"Unknown model architecture: {model_type}")
