"""Device helpers shared by model wrappers and attack runners."""

import torch


def get_device():
    """Return the preferred torch device for this machine."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_gpu_memory_gb(device_index=0):
    """Return total GPU memory in GB, or 0.0 when CUDA is unavailable."""
    if not torch.cuda.is_available():
        return 0.0
    props = torch.cuda.get_device_properties(device_index)
    return props.total_memory / 1024**3


def get_memory_based_batch_size(
    small_batch_size=16,
    large_batch_size=256,
    large_gpu_memory_gb=16,
):
    """Choose a conservative or large batch size based on available GPU memory."""
    gpu_memory_gb = get_gpu_memory_gb()
    if gpu_memory_gb >= large_gpu_memory_gb:
        return large_batch_size
    return small_batch_size
