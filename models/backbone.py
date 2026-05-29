import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class GeM(nn.Module):
    """Generalized Mean Pooling."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))
        return x.pow(1.0 / self.p)


class ResNet18Backbone(nn.Module):
    """ResNet-18 backbone with GeM pooling and BNNeck."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        try:
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            base = resnet18(weights=weights)
        except Exception:
            base = resnet18(pretrained=pretrained)

        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        self.gem = GeM()
        self.bnneck = nn.BatchNorm1d(512)
        self.bnneck.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.gem(x)
        x = x.view(x.size(0), -1)

        global_feat = x
        bn_feat = self.bnneck(global_feat)
        return global_feat, bn_feat
