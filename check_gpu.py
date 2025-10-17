import torch, torchvision, platform
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("torchvision:", torchvision.__version__)
print("is_cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("compute capability:", torch.cuda.get_device_capability(0))

print("To disable GPU and force CPU usage, write this in your terminal before running the main script:")
print('export CUDA_VISIBLE_DEVICES=\"\"')  # for Linux/Mac
print('set CUDA_VISIBLE_DEVICES=')        # for Windows
