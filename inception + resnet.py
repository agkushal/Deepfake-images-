import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score
import timm

#  Device
device = torch.device("cpu")

#  Paths
train_path = r"C:\Users\Bebop\Desktop\deepfakeAI\DATASETSMALL\train"
test_path  = r"C:\Users\Bebop\Desktop\deepfakeAI\DATASETSMALL\test"
csv_output_path = r"C:\Users\Bebop\Desktop\deepfakeAI\Results\inception_test_prob.csv"

os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)

#  Transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

#  Dataset / Dataloader
train_dataset = datasets.ImageFolder(train_path, transform=transform)
test_dataset  = datasets.ImageFolder(test_path,  transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=8, shuffle=False)

num_classes = len(train_dataset.classes)
print(f"🔹 Classes Detected: {train_dataset.classes}")

#  Ensemble (Inception v4 + ResNet50)
print("🔹 Loading Inception v4 and ResNet50 models...")

model_incep = timm.create_model('inception_v4', pretrained=True, num_classes=num_classes)
model_resnet = timm.create_model('resnet50', pretrained=True, num_classes=num_classes)

class EnsembleModel(nn.Module):
    def __init__(self, modelA, modelB):
        super().__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.fc = nn.Linear(num_classes * 2, num_classes)

    def forward(self, x):
        out1 = self.modelA(x)
        out2 = self.modelB(x)
        combined = torch.cat((out1, out2), dim=1)
        return self.fc(combined)

model = EnsembleModel(model_incep, model_resnet).to(device)

#  Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

#  Training
num_epochs = 3
print("\n Training Ensemble Model...\n")

for epoch in range(num_epochs):
    model.train()
    running_loss, all_preds, all_labels = 0.0, [], []

    for step, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        acc = accuracy_score(labels.cpu(), preds.cpu())

        running_loss += loss.item()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        if (step + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Step [{step+1}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f} | Step Acc: {acc*100:.2f}%")

    epoch_acc = accuracy_score(all_labels, all_preds)
    print(f"\n Epoch [{epoch+1}] Accuracy: {epoch_acc*100:.2f}%\n")

print("Training Complete!\n")

# Evaluation + CSV Export
model.eval()
all_image_names, all_true, all_pred, all_conf, all_prob_real, all_prob_fake = [], [], [], [], [], []

with torch.no_grad():
    for imgs, lbls in test_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        outputs = model(imgs)
        probs = F.softmax(outputs, dim=1)

        top_probs, preds = torch.max(probs, 1)

        for i in range(len(imgs)):
            img_path = test_loader.dataset.samples[len(all_image_names)][0]
            img_name = os.path.basename(img_path)

            # Handle binary or multi-class
            prob_real = probs[i][0].item() if num_classes >= 1 else 0.0
            prob_fake = probs[i][1].item() if num_classes > 1 else 0.0

            all_image_names.append(img_name)
            all_true.append(train_dataset.classes[lbls[i].item()])
            all_pred.append(train_dataset.classes[preds[i].item()])
            all_conf.append(top_probs[i].item())
            all_prob_real.append(prob_real)
            all_prob_fake.append(prob_fake)

#  Save to CSV
with open(csv_output_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Image_Name", "True_Label", "Predicted_Label",
                     "Confidence", "Prob_real", "Prob_fake"])
    for row in zip(all_image_names, all_true, all_pred,
                   all_conf, all_prob_real, all_prob_fake):
        writer.writerow(row)

print(f" Saved probability results to: {csv_output_path}")

#  Final Accuracy
test_acc = accuracy_score(all_true, all_pred)
print(f" Final Test Accuracy: {test_acc*100:.2f}%")
