import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
data_dir = r"C:\Users\User\Desktop\Medical-Image-Classification\data"

# 🔹 1. Config

batch_size = 16   # smaller batch for small dataset
num_epochs = 20   # train a bit longer but with early stopping
learning_rate = 0.0005
patience = 5      # for early stopping
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 🔹 2. Data transforms (heavier augmentations for TB dataset)
transform = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "test": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# 🔹 3. Datasets
image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), transform[x])
    for x in ["train", "val", "test"]
}
dataloaders = {
    x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True)
    for x in ["train", "val", "test"]
}
class_names = image_datasets["train"].classes
print("Classes:", class_names)

# 🔹 4. Model (Freeze most of ResNet18)
model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False   # freeze all layers

# unfreeze last layer
for param in model.layer4.parameters():
    param.requires_grad = True

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))  # new classifier
model = model.to(device)

# 🔹 Weighted loss (handles imbalance: Normal vs Pneumonia vs TB counts)
class_counts = [len(image_datasets["train"].targets) - image_datasets["train"].targets.count(i) for i in range(len(class_names))]
class_weights = torch.tensor(class_counts, dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 🔹 5. Training loop with Early Stopping
def train_model():
    best_acc = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val":
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), "best_model.pth")
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print("⏹️ Early stopping triggered")
                        return

    print("Training complete. Best val Acc: {:.4f}".format(best_acc))

# 🔹 6. Run Training
train_model()

# 🔹 7. Test Accuracy
def test_model():
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in dataloaders["test"]:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    print(f"✅ Test Accuracy: {100 * correct / total:.2f}%")

test_model()


