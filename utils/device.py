# utils/device.py
import torch

def get_device() -> str:
    """
    Return the best available device string:
      - 'cuda' if NVIDIA GPU is available
      - 'mps'  if Apple Silicon GPU is available
      - 'cpu'  otherwise
    """
    if torch.cuda.is_available():
        return "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"