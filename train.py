import torch
from torch import nn, optim
from torchvision import models, transforms
from datasets import load_dataset
from torch.utils.data import DataLoader, random_split, Dataset
from PIL import Image
import random
import numpy as np
import os

cache_dir = "/Users/kushdesai/waste-garbage-management-dataset"

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Material Classes

MATERIALS = ["plastic", "paper", "metal", "glass", "organic", "other"]
material_to_idx = {m: i for i, m in enumerate(MATERIALS)}

# Map original dataset labels to MATERIALS
label_map = {
    "metal": "metal",
    "glass": "glass",
    "biological": "organic",
    "paper": "paper",
    "cardboard": "paper",
    "plastic": "plastic",
    "battery": "other",
    "trash": "other",
    "shoes": "other",
    "clothes": "other"
}


# Load MobileNetV2 (Frozen)

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
model.classifier = nn.Identity()  # remove classifier
for param in model.parameters():
    param.requires_grad = False
model.eval()
model.to(device)


# Material Classifier

material_classifier = nn.Sequential(
    nn.Linear(1280, 128),
    nn.ReLU(),
    nn.Linear(128, len(MATERIALS))
)
material_classifier.train()
material_classifier.to(device)


# Image Transform + Augmentation

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),  # augmentation
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # augmentation
    transforms.RandomRotation(15),  # augmentation
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Custom Dataset

class WasteDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = Image.open(item["image"]["path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        material_label = label_map[item["label"]]
        label_idx = material_to_idx[material_label]
        return img, label_idx


# Load Dataset & Split

full_dataset = load_dataset(
    "omasteam/waste-garbage-management-dataset",
    split="train",
    cache_dir=cache_dir
)
dataset = WasteDataset(full_dataset, transform=transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# Loss & Optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(material_classifier.parameters(), lr=1e-3)


# Training Loop (with best model saving)

EPOCHS = 5  # can increase later
best_val_acc = 0

for epoch in range(EPOCHS):
    # --- Training ---
    material_classifier.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        # Feature extraction (frozen)
        with torch.no_grad():
            features = model(images)

        # Classifier forward
        outputs = material_classifier(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total

    # --- Validation ---
    material_classifier.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            features = model(images)
            outputs = material_classifier(features)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    val_acc = 100 * val_correct / val_total

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss:.3f} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(material_classifier.state_dict(), "best_material_classifier.pth")
        print("Best model saved!")

print("Training finished. Best model accuracy:", best_val_acc)
