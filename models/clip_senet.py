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
    """Main model combining ResNet-18, CLIP SEM, AFEM, and fusion head."""

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
        return (x - mean) / std

    def forward(self, images: torch.Tensor, return_logits: bool = True):
        cnn_in = self._normalize(images, self.imagenet_mean, self.imagenet_std)
        global_feat, bn_feat = self.backbone(cnn_in)

        if self.use_sem:
            clip_in = self._normalize(images, self.clip_mean, self.clip_std)
            clip_feat = self.sem(clip_in)
            if self.use_afem:
                clip_feat = self.afem(clip_feat)
        else:
            clip_feat = torch.zeros_like(bn_feat)

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
