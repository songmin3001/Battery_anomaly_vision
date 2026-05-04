"""
train.py
EfficientNet-B3 학습 스크립트
- 클래스 불균형 보정 / Early Stopping / Best 모델 자동 저장
"""

import argparse
import json
import time
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import get_dataloaders, CLASS_NAMES, NUM_CLASSES, BatteryDataset
from model import build_model, count_params


def compute_class_weights(data_dir, device):
    ds = BatteryDataset(data_dir)
    counts = Counter(label for _, label, _ in ds)
    total = sum(counts.values())
    weights = torch.tensor(
        [total / (NUM_CLASSES * counts[i]) for i in range(NUM_CLASSES)],
        dtype=torch.float32
    )
    print("[Class Weights]", {CLASS_NAMES[i]: f"{weights[i]:.3f}" for i in range(NUM_CLASSES)})
    return weights.to(device)


class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels, _ in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels, _ in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    train_loader, val_loader, _ = get_dataloaders(
        args.data_dir, args.batch_size, num_workers=args.num_workers, seed=args.seed
    )

    model = build_model(device)
    print(count_params(model))

    class_weights = compute_class_weights(args.data_dir, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "best_model.pth"
    early_stop = EarlyStopping(patience=args.patience)
    best_val_acc = 0.0
    history = []

    print(f"\n{'Epoch':>6} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>10} {'Val Acc':>9}")
    print("─" * 55)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"{epoch:>6} {train_loss:>11.4f} {train_acc:>9.4f} "
              f"{val_loss:>10.4f} {val_acc:>9.4f}  ({time.time()-t0:.1f}s)")

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_acc": val_acc, "val_loss": val_loss}, best_path)
            print(f"  → Best 저장 (val_acc={val_acc:.4f})")

        if early_stop(val_loss):
            print(f"\n[Early Stopping] {epoch} 에폭에서 종료")
            break

    print(f"\n학습 완료. Best val_acc={best_val_acc:.4f}")
    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--save_dir", default="../runs")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    train(parser.parse_args())
    
