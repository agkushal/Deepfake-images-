import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import ViTForImageClassification, ViTFeatureExtractor, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
from torch.nn.functional import softmax

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_path = r"C:\Users\Bebop\Desktop\deepfakeAI\DATASETSMALL\train"
test_path  = r"C:\Users\Bebop\Desktop\deepfakeAI\DATASETSMALL\test"

dataset = load_dataset(
    "imagefolder",
    data_files={
        "train": f"{train_path}/**",
        "test": f"{test_path}/**",
    },
)

split_dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
dataset["train"] = split_dataset["train"]
dataset["validation"] = split_dataset["test"]

print(f"Training samples: {len(dataset['train'])}")
print(f"Validation samples: {len(dataset['validation'])}")
print(f"Testing samples: {len(dataset['test'])}")

feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

def transform(example_batch):
    inputs = feature_extractor([x for x in example_batch['image']], return_tensors='pt')
    inputs['labels'] = example_batch['label']
    return inputs

dataset = dataset.with_transform(transform)

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }

num_labels = len(dataset["train"].features["label"].names)
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=num_labels,
    ignore_mismatched_sizes=True
)
model.to(device)

args = TrainingArguments(
    output_dir="./vit-deepfake-output",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    remove_unused_columns=False,
    dataloader_pin_memory=False
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=collate_fn,
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
)

trainer.train()

metrics = trainer.evaluate(dataset["test"])
print("\nEvaluation Results:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

predictions = trainer.predict(dataset["test"])
y_true = predictions.label_ids
y_pred = predictions.predictions.argmax(-1)

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dataset["train"].features["label"].names)
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix - Deepfake Detection")
plt.show()

probs = softmax(torch.tensor(predictions.predictions).to(device), dim=1).cpu().numpy()
y_true = predictions.label_ids
y_pred = predictions.predictions.argmax(-1)
label_names = dataset["train"].features["label"].names

test_plain = dataset["test"].with_transform(lambda x: x)
filepaths, true_labels_text = [], []

for i in range(len(test_plain)):
    ex = test_plain[i]
    img = ex["image"]
    fp = getattr(img, "filename", "") if hasattr(img, "filename") else f"test_{i:05d}"
    filepaths.append(fp)
    true_labels_text.append(label_names[ex["label"]])

df = pd.DataFrame({
    "Image_Name": [os.path.basename(f).lower() for f in filepaths],
    "True_Label": true_labels_text,
    "Predicted_Label": [label_names[i] for i in y_pred],
    "Confidence": probs.max(axis=1),
    "Prob_real": probs[:, label_names.index("real")],
    "Prob_fake": probs[:, label_names.index("fake")]
})

os.makedirs(r"C:\Users\Bebop\Desktop\deepfakeAI\Results", exist_ok=True)
out_csv = r"C:\Users\Bebop\Desktop\deepfakeAI\Results\vit_test_prob.csv"
df.to_csv(out_csv, index=False, encoding="utf-8")
print(f"\nVIT probability scores saved at: {out_csv}")
print(df.head(10).to_string(index=False))
