
from pathlib import Path

from models.Gen_Model_Wrapper import GenomeGenerativeModelWrapper


def create_model_wrapper(
    file_name: str,
    device=None,
    generation_batch_size=None,
) -> 'GenomeGenerativeModelWrapper':
    model_stem = Path(file_name).stem
    model_type = model_stem.split('_')[0].upper()
    
    if model_stem.startswith("PCA_DM"):
        from models.PCA_DM.generative_PCA_DM import PCA_DM_generative

        kwargs = {"model_path": file_name, "device": device}
        if generation_batch_size is not None:
            kwargs["generation_batch_size"] = generation_batch_size
        return PCA_DM_generative(**kwargs)
    elif model_stem.startswith("AC_GAN"):
        from models.AC_GAN.generative_ACGAN import ACGAN_generative

        kwargs = {"model_path": file_name}
        if generation_batch_size is not None:
            kwargs["generation_batch_size"] = generation_batch_size
        return ACGAN_generative(**kwargs)
    elif model_type == "VAE":
        from models.VAE.generative_VAE import VAE_generative

        kwargs = {"model_path": file_name, "device": device}
        if generation_batch_size is not None:
            kwargs["generation_batch_size"] = generation_batch_size
        return VAE_generative(**kwargs)
    elif model_type == "WGAN":
        from models.WGAN.generative_WGAN import WGAN_generative

        kwargs = {"model_path": file_name, "device": device}
        if generation_batch_size is not None:
            kwargs["generation_batch_size"] = generation_batch_size
        return WGAN_generative(**kwargs)
    else:
        raise ValueError(f"Unknown model architecture: {model_type}")
