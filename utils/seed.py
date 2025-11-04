# utils/seed.py
import torch

def set_global_seeds(seed: int, deterministic: bool = True) -> None:
    """
    Set reproducible seeds for CPU, CUDA, or MPS.
    On macOS (MPS backend) only torch.manual_seed() is needed.
    On CUDA devices, also sets cuDNN deterministic flags.
    """
    torch.manual_seed(seed)

    # CUDA-specific determinism
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # MPS backend does not require special handling for seeds
    # (torch.manual_seed already covers CPU + MPS streams)