
from models.Gen_Model_Wrapper import GenomeGenerativeModelWrapper
from models.VAE.generative_VAE import VAE_generative
from pathlib import Path

def create_model_wrapper(file_name: str) -> 'GenomeGenerativeModelWrapper':
    model_type = Path(file_name).stem.split('_')[0].upper()
    
    if model_type == "VAE":
        return VAE_generative(model_path=file_name)
    else:
        raise ValueError(f"Unknown model architecture: {model_type}")
