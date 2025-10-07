import torch
import torch.nn as nn
import torchvision.models as M

class ViTBackbone(nn.Module):
    """
    Vision Transformer (ViT-B/16) backbone.
    Extracts 768-d CLS token feature vector from final transformer block.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = M.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        vit = M.vit_b_16(weights=weights)
        vit.heads.head = nn.Identity()  # remove classifier head
        self.vit = vit
        self.vit_last = vit.encoder.layers[-1]
        self.out_dim = 768

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(x)


class ViTClassifier(nn.Module):
    """
    ViT-B/16 classifier for MRI classes.
    """
    def __init__(self, num_classes: int = 3, pretrained: bool = True):
        super().__init__()
        self.backbone = ViTBackbone(pretrained=pretrained)
        self.head = nn.Linear(self.backbone.out_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.head(feats)
