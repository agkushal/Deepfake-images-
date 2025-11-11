import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

#  Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Using device: {device}")

#  Paths
train_path = r"C:\Users\Bebop\Desktop\deepfakeAI\DATASETSMALL\train"
test_path  = r"C:\Users\Bebop\Desktop\deepfakeAI\DATASETSMALL\test"

# CSV Save Location
csv_output_path = r"C:\Users\Bebop\Desktop\deepfakeAI\Results\google_test_prob.csv"

# Transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # GoogLeNet input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

#  Dataset & Dataloaders
train_dataset = datasets.ImageFolder(train_path, transform=transform)
test_dataset = datasets.ImageFolder(test_path, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False)

class_names = train_dataset.classes
num_classes = len(class_names)
print(f"🔹 Classes detected: {class_names}")

#  Load GoogLeNet Model (Pretrained)
model = models.googlenet(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.to(device)

#  Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

#  Training
num_epochs = 10
print("\n🚀 Training GoogLeNet model...\n")

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for step, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        probs = F.softmax(outputs, dim=1)
        conf, preds = torch.max(probs, 1)
        acc = accuracy_score(labels.cpu(), preds.cpu())

        total_loss += loss.item()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        print(f"Epoch [{epoch+1}/{num_epochs}] Step [{step+1}/{len(train_loader)}] "
              f"Loss: {loss.item():.4f} | Step Acc: {acc*100:.2f}% | "
              f"Avg Conf: {conf.mean().item()*100:.2f}%")

    epoch_acc = accuracy_score(all_labels, all_preds)
    print(f"\n Epoch [{epoch+1}] Accuracy: {epoch_acc*100:.2f}% | "
          f"Avg Loss: {total_loss/len(train_loader):.4f}\n")

print(" Training complete!\n")

# Evaluation + Save CSV of Probability Scores (Unified Format)
import os, csv, torch.nn.functional as F
import pandas as pd

model.eval()
results = []

print(f"\nSaving probability scores to: {csv_output_path}\n")

with torch.no_grad():
    all_preds, all_labels = [], []  # <-- Added for final accuracy
    for i, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        conf, preds = torch.max(probs, 1)

        all_preds.extend(preds.cpu().numpy())   # <-- Added
        all_labels.extend(labels.cpu().numpy()) # <-- Added

        for j in range(images.size(0)):
            img_idx = i * test_loader.batch_size + j
            img_path, _ = test_dataset.samples[img_idx]

            results.append({
                "Image_Name": os.path.basename(img_path).lower(),
                "True_Label": class_names[labels[j].item()],
                "Predicted_Label": class_names[preds[j].item()],
                "Confidence": conf[j].item(),
                "Prob_real": probs[j][class_names.index("real")].item(),
                "Prob_fake": probs[j][class_names.index("fake")].item()
            })

#  Convert to DataFrame and Save CSV (matching ViT format)
os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
df = pd.DataFrame(results)
df.to_csv(csv_output_path, index=False, encoding="utf-8")

print(f" Probability scores saved successfully at:\n➡ {csv_output_path}")
print(df.head(10).to_string(index=False))

#  Final Test Accuracy
final_acc = accuracy_score(all_labels, all_preds)
print(f"\n Final Test Accuracy: {final_acc*100:.2f}%\n")
