import torch
import torchvision
import PIL
import numpy
import cv2

print("PyTorch version:", torch.__version__)
print("torchvision version:", torchvision.__version__)
print("Pillow version:", PIL.__version__)
print("NumPy version:", numpy.__version__)
print("OpenCV version:", cv2.__version__)

cuda_available = torch.cuda.is_available()
print("CUDA available:", cuda_available)
if cuda_available:
    print("CUDA version:", torch.version.cuda)
