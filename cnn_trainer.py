#  475 training images and 132 validation images.

import os
from pathlib import Path
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ======================================================
# CONFIGURATION
# ======================================================
DATA_DIR = Path(r"\dataset\crops")
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "valid"
MODEL_SAVE_PATH = Path(r"\chess_cnn.pth")

BATCH_SIZE = 16
EPOCHS = 20
LR = 0.001
IMG_SIZE = 64
# ======================================================

# Image augmentations / normalization
transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_val = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# Datasets
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform_train)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"✅ Found {len(train_dataset)} training images and {len(val_dataset)} validation images.")
print(f"Classes: {train_dataset.classes}")

# ======================================================
# MODEL DEFINITION
# ======================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Use a small pretrained CNN (transfer learning)
model = models.resnet18(weights="IMAGENET1K_V1")
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ======================================================
# TRAINING LOOP
# ======================================================
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()

    train_acc = correct / len(train_dataset)
    val_correct = 0
    val_loss = 0

    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = outputs.argmax(dim=1)
            val_correct += (preds == labels).sum().item()

    val_acc = val_correct / len(val_dataset)
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% | Loss: {val_loss/len(val_loader):.4f}")

# ======================================================
# SAVE MODEL
# ======================================================
torch.save({
    "model_state": model.state_dict(),
    "class_names": train_dataset.classes
}, MODEL_SAVE_PATH)

print(f"\n✅ Training complete! Model saved to {MODEL_SAVE_PATH}")
