import torch
import torch.nn as nn

from models.backbone import ResNet18Backbone
from models.sem import CLIPSEM
from models.afem import AFEM

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class ClipSENet(nn.Module):
    """CLIP-SE Lite: Main model combining visual and semantic branches for Vehicle ReID.

    Architecture overview:
      1. **Visual branch** (ResNet-18): Extracts 512-dim geometric/texture features.
      2. **Semantic branch** (CLIP ViT-B/16 SEM): Extracts 512-dim semantic features
         (color, vehicle type, high-level concepts).
      3. **AFEM**: Refines CLIP features via group-wise recalibration.
      4. **Fusion head**: Concatenates visual + semantic features (1024-dim),
         projects to 512-dim with BN, and feeds into the classifier.

    The model applies separate normalization for each branch: ImageNet
    statistics for ResNet-18 and CLIP statistics for the ViT encoder.
    Normalization buffers are registered (non-persistent) so they move
    to the correct device automatically.

    Args:
        num_classes: Number of vehicle identities in the training set.
        pretrained: Use ImageNet-pretrained ResNet-18 weights.
        use_sem: Enable the CLIP semantic branch.
        use_afem: Enable the AFEM feature enhancement module.
        afem_groups: Number of groups (G) for AFEM recalibration.
        clip_img_size: Input resolution for the CLIP branch.
        clip_patch_size: Patch size for ViT (16 for ViT-B/16).
        unfreeze_last_blocks: Number of final ViT blocks to fine-tune.
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        use_sem: bool = True,
        use_afem: bool = True,
        afem_groups: int = 32,
        clip_img_size: int = 320,
        clip_patch_size: int = 16,
        unfreeze_last_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.backbone = ResNet18Backbone(pretrained=pretrained)
        self.use_sem = use_sem
        self.use_afem = use_afem

        if self.use_sem:
            self.sem = CLIPSEM(
                img_size=clip_img_size,
                patch_size=clip_patch_size,
                unfreeze_last_blocks=unfreeze_last_blocks,
            )
        else:
            self.sem = None

        if self.use_afem:
            self.afem = AFEM(in_channels=512, num_groups=afem_groups)
        else:
            self.afem = None

        self.concat_fc = nn.Linear(1024, 512, bias=False)
        self.concat_norm = nn.BatchNorm1d(512)
        self.classifier = nn.Linear(512, num_classes, bias=False)

        self.register_buffer(
            "imagenet_mean",
            torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor(IMAGENET_STD).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "clip_mean",
            torch.tensor(CLIP_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "clip_std",
            torch.tensor(CLIP_STD).view(1, 3, 1, 1),
            persistent=False,
        )

    def _normalize(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Apply channel-wise normalization to a batch of images.

        Args:
            x: Raw images of shape ``(B, 3, H, W)`` in [0, 1] range.
            mean: Per-channel mean as ``(1, 3, 1, 1)`` tensor.
            std: Per-channel std as ``(1, 3, 1, 1)`` tensor.

        Returns:
            Normalized images of same shape.
        """
        return (x - mean) / std

    def forward(self, images: torch.Tensor, return_logits: bool = True):
        """Run the full dual-branch forward pass.

        Args:
            images: Raw images of shape ``(B, 3, H, W)`` in [0, 1] range.
                The model handles normalization internally (ImageNet for
                ResNet, CLIP stats for ViT).
            return_logits: If True, compute classification logits for
                Cross-Entropy loss. Set to False during inference.

        Returns:
            Dict with keys:
              - ``logits``: Classification logits ``(B, num_classes)`` or None.
              - ``fusion_feat``: Final fused features ``(B, 512)`` used for
                retrieval and metric loss.
              - ``global_feat``: Raw ResNet features before BNNeck ``(B, 512)``.
              - ``bn_feat``: BNNeck-normalized ResNet features ``(B, 512)``.
              - ``clip_feat``: CLIP/AFEM semantic features ``(B, 512)``.
        """
        cnn_in = self._normalize(images, self.imagenet_mean, self.imagenet_std)
        global_feat, bn_feat = self.backbone(cnn_in)

        if not self.use_sem:
            logits = self.classifier(bn_feat) if return_logits else None
            return {
                "logits": logits,
                "fusion_feat": bn_feat,
                "global_feat": global_feat,
                "bn_feat": bn_feat,
                "clip_feat": torch.zeros_like(bn_feat),
            }

        clip_in = self._normalize(images, self.clip_mean, self.clip_std)
        clip_feat = self.sem(clip_in)
        if self.use_afem:
            clip_feat = self.afem(clip_feat)

        fusion = torch.cat([bn_feat, clip_feat], dim=1)
        fusion = self.concat_norm(self.concat_fc(fusion))

        logits = self.classifier(fusion) if return_logits else None

        return {
            "logits": logits,
            "fusion_feat": fusion,
            "global_feat": global_feat,
            "bn_feat": bn_feat,
            "clip_feat": clip_feat,
        }
