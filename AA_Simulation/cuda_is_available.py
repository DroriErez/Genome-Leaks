import torch

print("CUDA available:", torch.cuda.is_available())
print("Devices:", torch.cuda.device_count())
print(torch.__version__)
print(torch.version.cuda)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))