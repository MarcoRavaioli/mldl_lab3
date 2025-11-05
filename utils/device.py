import torch

def get_device() -> str:
    """
    Pick best device:
      - 'cuda' if available
      - 'mps' on Apple Silicon
      - 'cpu' otherwise
    """
    if torch.cuda.is_available():
        return "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"