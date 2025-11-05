from __future__ import annotations
import argparse
import torch
import torch.nn as nn

from data.tiny_imagenet import build_tiny_imagenet_loaders
from models.custom_net import CustomNet
from utils.device import get_device

@torch.no_grad()
def eval_loop(model: nn.Module, loader, device: str):
    model.eval()
    total, correct = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    print(f"Accuracy: {100.0 * correct / total:.2f}%")

def parse_args():
    p = argparse.ArgumentParser("Evaluate CustomNet on TinyImageNet")
    p.add_argument("--datasets-root", type=str, default="datasets")
    p.add_argument("--weights", type=str, default="checkpoints/lab2_best.ckpt")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--img-size", type=int, default=224)
    return p.parse_args()

def main():
    args = parse_args()
    device = get_device()

    _, val_loader, num_classes = build_tiny_imagenet_loaders(
        datasets_root=args.datasets_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        pin_memory=(device == "cuda"),
    )
    model = CustomNet(num_classes=num_classes).to(device)
    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded: {args.weights} (epoch={ckpt.get('epoch')}, val_top1={ckpt.get('val_top1')})")
    eval_loop(model, val_loader, device)

if __name__ == "__main__":
    main()