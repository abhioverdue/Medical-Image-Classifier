import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Always resolve project root dynamically
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
default_data_dir = os.path.join(project_root, "data")


def get_dataloaders(data_dir=default_data_dir, batch_size=32, img_size=224):
    transform_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229])
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform_train)
    val_dataset   = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=transform_val)
    test_dataset  = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=transform_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_dataset.classes

