import torch
import torch.nn as nn
import torchvision.models as M

class ResNetBackbone(nn.Module):
    """
    ResNet-50 backbone for MRI image feature extraction.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = M.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        resnet = M.resnet50(weights=weights)
        self.features = nn.Sequential(*list(resnet.children())[:-2])  # up to layer4
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = 2048
        self.layer4 = resnet.layer4[-1]  # for Grad-CAM targeting

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return torch.flatten(x, 1)


class ResNetClassifier(nn.Module):
    """
    ResNet50 classifier for MRI dataset.
    """
    def __init__(self, num_classes: int = 3, pretrained: bool = True):
        super().__init__()
        self.backbone = ResNetBackbone(pretrained=pretrained)
        self.head = nn.Linear(self.backbone.out_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.head(feats)
