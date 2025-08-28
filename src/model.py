import torch.nn as nn
import torchvision.models as models

def build_model(num_classes=3, pretrained=True):
    """
    Build a ResNet18-based model for medical image classification.

    Args:
        num_classes (int): Number of output classes (default=3 -> Normal, Pneumonia, Tuberculosis).
        pretrained (bool): Whether to use pretrained ImageNet weights.

    Returns:
        model (torch.nn.Module): Modified ResNet18 model.
    """
    model = models.resnet18(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )

    return model
