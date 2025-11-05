from __future__ import annotations
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from data.tiny_imagenet import build_tiny_imagenet_loaders
from models.custom_net import CustomNet
from utils.device import get_device
from utils.seed import set_global_seeds

def train_one_epoch(model, loader, criterion, optimizer, device: str):
    model.train()
    total_loss, total_top1, n = 0.0, 0.0, 0
    for i, (images, targets) in enumerate(loader):
        if i == 0: print("  got first batch")
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        preds = logits.argmax(1)
        bs = targets.size(0)
        total_loss += loss.item() * bs
        total_top1 += (preds == targets).float().sum().item()
        n += bs
        if (i+1) % 50 == 0: print(f"  step {i+1}/{len(loader)}")
    return total_loss / n, (100.0 * total_top1 / n)

@torch.no_grad()
def evaluate(model, loader, criterion, device: str):
    model.eval()
    total_loss, total_top1, n = 0.0, 0.0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        preds = logits.argmax(1)
        bs = targets.size(0)
        total_loss += loss.item() * bs
        total_top1 += (preds == targets).float().sum().item()
        n += bs
    return total_loss / n, (100.0 * total_top1 / n)

def parse_args():
    p = argparse.ArgumentParser("Train CustomNet on TinyImageNet (Lab 2 -> Lab 3 skeleton)")
    p.add_argument("--datasets-root", type=str, default="datasets")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-path", type=str, default="checkpoints/lab2_best.ckpt")
    return p.parse_args()

def main():
    args = parse_args()
    set_global_seeds(args.seed)
    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader, num_classes = build_tiny_imagenet_loaders(
        datasets_root=args.datasets_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        pin_memory=(device == "cuda"),
    )
    print(f"Classes: {num_classes} | Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

    model = CustomNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    # make sure checkpoints dir exists
    os.makedirs("checkpoints", exist_ok=True)

    best_top1 = -1.0
    for epoch in range(1, args.epochs + 1):
        print(f"[Epoch {epoch:02d}] starting…")
        tr_loss, tr_top1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_top1 = evaluate(model, val_loader, criterion, device)
        print(f"[Epoch {epoch:02d}] train_loss={tr_loss:.4f} top1={tr_top1:.2f} | "
              f"val_loss={va_loss:.4f} top1={va_top1:.2f}")

        if va_top1 > best_top1:
            best_top1 = va_top1
            torch.save(
                {"model_state": model.state_dict(), "epoch": epoch, "val_top1": va_top1},
                args.save_path,
            )
            print(f"  ↳ Saved best checkpoint to {args.save_path}")

if __name__ == "__main__":
    import os
    main()