
# Deepfake CNN 

import os
from pathlib import Path
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.amp import autocast, GradScaler

import matplotlib.pyplot as plt

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(42)


# 1. Custom Dataset (Loads + Preprocesses Images in RAM)

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root = Path(root_dir)
        self.transform = transform

        self.samples = []
        self.class_to_idx = {"real": 0, "fake": 1}

        for cls in ["real", "fake"]:
            folder = self.root / cls
            if not folder.exists():
                continue
            for img_name in os.listdir(folder):
                path = folder / img_name
                self.samples.append((path, self.class_to_idx[cls]))

        print(f"Loaded {len(self.samples)} images from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label

# CNN Model (Simple + Strong Generalization)

class DeepFakeCNNv2(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),   

            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),   

            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),   

            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


#  Training Function

def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler):
    model.train()
    losses = []
    preds_all = []
    labels_all = []

    pbar = tqdm(loader, desc="Train")

    for imgs, labels in pbar:
        imgs = imgs.to(device)
        labels = torch.tensor(labels).to(device)

        with autocast("cuda"):
            logits = model(imgs)
            loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())
        preds_all.extend(logits.argmax(1).detach().cpu().numpy())
        labels_all.extend(labels.cpu().numpy())

        pbar.set_postfix(loss=f"{np.mean(losses):.4f}")

    acc = accuracy_score(labels_all, preds_all)
    return np.mean(losses), acc


#  Evaluation Function

def evaluate(model, loader, device):
    model.eval()
    preds_all = []
    labels_all = []

    pbar = tqdm(loader, desc="Eval")

    with torch.no_grad():
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            logits = model(imgs)

            preds_all.extend(logits.argmax(1).cpu().numpy())
            labels_all.extend(labels)

    return accuracy_score(labels_all, preds_all)


#  Main Training Script

def main():
    DATASET_PATH = Path("C:\Users\Bebop\Desktop\deepfakeAI\Dataset")

    train_dir = DATASET_PATH / "train"
    test_dir  = DATASET_PATH / "test"

    train_tf = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2,0.2,0.15),
        transforms.ToTensor()
    ])

    test_tf = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor()
    ])

    train_ds = DeepfakeDataset(train_dir, transform=train_tf)
    test_ds  = DeepfakeDataset(test_dir, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = DeepFakeCNNv2().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    scaler = GradScaler("cuda")

    EPOCHS = 15


    history_train_loss = []
    history_train_acc = []
    history_val_acc = []

    best_train_acc = 0
    best_val_acc = 0


    for epoch in range(1, EPOCHS+1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler)
        val_acc = evaluate(model, test_loader, device)

        history_train_loss.append(train_loss)
        history_train_acc.append(train_acc)
        history_val_acc.append(val_acc)

        if train_acc > best_train_acc:
            best_train_acc = train_acc
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        print(f"\nEpoch {epoch}/{EPOCHS} | Train Acc = {train_acc:.4f} | Val Acc = {val_acc:.4f} | Time = {time.time()-t0:.1f}s")

    print("\nTraining Complete.")
    print("Best Training Accuracy:", best_train_acc)
    print("Best Validation Accuracy:", best_val_acc)

    # Plot graphs
   
    plt.figure(figsize=(10,5))
    plt.plot(history_train_loss)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    plt.figure(figsize=(10,5))
    plt.plot(history_train_acc, label="Train Acc")
    plt.plot(history_val_acc, label="Validation Acc")
    plt.title("Accuracy Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
