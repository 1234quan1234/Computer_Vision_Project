import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class GeM(nn.Module):
    """Generalized Mean (GeM) Pooling.

    Replaces standard average pooling with a learnable pooling exponent ``p``.
    When ``p=1`` it behaves like average pooling; as ``p`` increases it
    approaches max pooling. The learned ``p`` allows the network to
    adaptively balance between average and max pooling, which often
    improves retrieval performance in ReID tasks.

    Args:
        p: Initial pooling exponent (default: 3.0, learned during training).
        eps: Small constant to clamp input values and avoid numerical issues.
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GeM pooling over spatial dimensions (H, W).

        Args:
            x: Feature map of shape ``(B, C, H, W)``.

        Returns:
            Pooled tensor of shape ``(B, C, 1, 1)``.
        """
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))
        return x.pow(1.0 / self.p)


class ResNet18Backbone(nn.Module):
    """ResNet-18 visual backbone with GeM pooling and BNNeck.

    Extracts 512-dim visual features from input images. The backbone
    outputs two variants of the feature vector:
      - ``global_feat``: Raw GeM-pooled features (used for metric losses).
      - ``bn_feat``: BatchNorm-normalized features (used for classification
        and fusion with CLIP semantic features).

    BNNeck (Batch Normalization Neck) decouples the feature spaces for
    ID classification and metric learning, following the technique from
    "Bag of Tricks for Re-ID" (Luo et al., 2019).

    Args:
        pretrained: If True, initialize with ImageNet-1K pretrained weights.
    """

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
        """Extract visual features from a batch of normalized images.

        Args:
            x: ImageNet-normalized images of shape ``(B, 3, H, W)``.

        Returns:
            Tuple of ``(global_feat, bn_feat)`` where both are ``(B, 512)``.
        """
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
