# scripts/lab1_ex2.py
from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

from data.tiny_imagenet import (
    build_tiny_imagenet_loaders,
    IMAGENET_MEAN, IMAGENET_STD
)
from utils.device import get_device
from utils.seed import set_global_seeds

def denormalize(t: torch.Tensor) -> np.ndarray:
    img = t.detach().cpu().numpy().transpose(1, 2, 0)
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std  = np.array(IMAGENET_STD,  dtype=np.float32)
    img = img * std + mean
    return np.clip(img, 0.0, 1.0)

def parse_args():
    p = argparse.ArgumentParser("Lab1-Ex02 — DataLoader & Visualization")
    p.add_argument("--data-root", type=str, default="data")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--img-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()

def main():
    args = parse_args()
    set_global_seeds(args.seed)

    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader, num_classes = build_tiny_imagenet_loaders(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        pin_memory=(device == "cuda")
    )

    n_train = len(train_loader.dataset)
    n_val   = len(val_loader .dataset)
    print(f"Number of classes: {num_classes}")
    print(f"Train samples: {n_train} | Val samples: {n_val}")

    # Collect one example for each of 10 classes
    to_show = 10
    sampled = {}
    for images, targets in train_loader:
        for img, cls in zip(images, targets):
            c = int(cls.item())
            if c not in sampled:
                sampled[c] = img
                if len(sampled) >= to_show:
                    break
        if len(sampled) >= to_show:
            break

    # Plot grid
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for ax, (cls_idx, img) in zip(axes.flatten(), sorted(sampled.items(), key=lambda x: x[0])):
        ax.imshow(denormalize(img))
        ax.set_title(f"class {cls_idx}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()