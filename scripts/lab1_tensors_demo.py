# scripts/lab1_tensors_demo.py
# MLDL_Lab01_Ex01 — PyTorch fundamentals demo (cross-platform: CPU/CUDA/MPS)

import torch
from utils.device import get_device
from utils.seed import set_global_seeds

def main():
    device = get_device()
    print(f"Using device: {device}")

    # 2) Random (7,7)
    t2 = torch.rand((7, 7), device=device)
    print("\n[2] Random (7,7) tensor:\n", t2, t2.shape)

    # 3) Matmul with (1,7)^T -> (7,1)
    t3_rhs = torch.rand((1, 7), device=device)
    t3_out = t2 @ t3_rhs.t()
    print("\n[3] Matmul result and shape:\n", t3_out, t3_out.shape)

    # 4) Seed=0, redo (2) & (3)
    set_global_seeds(0)
    A = torch.rand((7, 7), device=device)
    B = torch.rand((1, 7), device=device)
    out_4 = A @ B.t()
    print("\n[4] Seed=0 -> reproducible result & shape:\n", out_4, out_4.shape)

    # 5) Display GPU backend info
    if device == "cuda":
        print("\n[5] CUDA backend active.")
    elif device == "mps":
        print("\n[5] MPS backend active (Apple Silicon).")
    else:
        print("\n[5] Running on CPU.")

    # 6) Two random (2,3) tensors, seed=1234
    set_global_seeds(1234)
    tensor_A = torch.rand((2, 3), device=device)
    tensor_B = torch.rand((2, 3), device=device)
    print(f"\n[6] Device: {device}")
    print(tensor_A, "\n", tensor_B)

    # 7) Matmul (2,3) @ (3,2)
    out_7 = tensor_A @ tensor_B.t()
    print("\n[7] Matmul(A, B^T) and shape:\n", out_7, out_7.shape)

    # 8) Max / Min
    print("\n[8] Max / Min of output(7):\n", out_7.max(), out_7.min())

    # 9) Argmax / Argmin
    argmax_idx = out_7.argmax()
    argmin_idx = out_7.argmin()
    argmax_rc = divmod(argmax_idx.item(), out_7.size(1))
    argmin_rc = divmod(argmin_idx.item(), out_7.size(1))
    print("\n[9] Argmax / Argmin (flat):", argmax_idx.item(), argmin_idx.item())
    print("    Argmax (row, col):", argmax_rc, " | Argmin (row, col):", argmin_rc)

    # 10) (1,1,1,10) -> squeeze -> (10)
    set_global_seeds(7)
    t10 = torch.rand((1, 1, 1, 10), device=device)
    t10_squeezed = t10.squeeze()
    print("\n[10] Original and squeezed tensors + shapes:")
    print(t10, t10.shape)
    print(t10_squeezed, t10_squeezed.shape)

if __name__ == "__main__":
    main()