import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import clip
except ImportError as exc:
    raise ImportError(
        "OpenAI CLIP is required. Install with: pip install git+https://github.com/openai/CLIP.git"
    ) from exc


def interpolate_pos_encoding(clip_model, img_size=320, patch_size=16):
    pos_embed = clip_model.visual.positional_embedding # [197, 768]
    cls_token = pos_embed[0:1, :] # [1, 768]
    patch_embed = pos_embed[1:, :] # [196, 768]
    
    orig_size = int(math.sqrt(patch_embed.shape[0])) # 14
    new_size = img_size // patch_size # 20
    
    # Reshape to [1, 768, 14, 14]
    patch_embed = patch_embed.reshape(1, orig_size, orig_size, 768).permute(0, 3, 1, 2)
    # Interpolate to [1, 768, 20, 20]
    patch_embed = F.interpolate(patch_embed, size=(new_size, new_size), mode='bicubic', align_corners=False)
    # Reshape back to [1, 400, 768]
    patch_embed = patch_embed.permute(0, 2, 3, 1).reshape(1, new_size*new_size, 768)
    
    # Concatenate CLS token -> [1, 401, 768]
    new_pos_embed = torch.cat([cls_token.unsqueeze(0), patch_embed], dim=1)
    return new_pos_embed


class CLIPSEM(nn.Module):
    """CLIP ViT-B/16 wrapper for semantic extraction at 320x320."""

    def __init__(
        self,
        model_name: str = "ViT-B/16",
        img_size: int = 320,
        patch_size: int = 16,
        unfreeze_last_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size

        clip_model, _ = clip.load(model_name, device="cpu", jit=False)
        self.clip_model = clip_model
        self.visual = clip_model.visual

        # Freeze all visual params by default, then unfreeze the last N blocks.
        for param in self.visual.parameters():
            param.requires_grad = False

        if unfreeze_last_blocks > 0:
            blocks = self.visual.transformer.resblocks
            for param in blocks[-unfreeze_last_blocks:].parameters():
                param.requires_grad = True

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        visual = self.visual

        x = visual.conv1(images)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)

        cls_token = visual.class_embedding.to(x.dtype)
        cls_token = cls_token.unsqueeze(0).unsqueeze(0).expand(x.shape[0], 1, -1)
        x = torch.cat([cls_token, x], dim=1)

        pos_embed = interpolate_pos_encoding(self.clip_model, img_size=self.img_size, patch_size=self.patch_size)
        pos_embed = pos_embed.to(dtype=x.dtype, device=x.device)
        x = x + pos_embed

        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)

        x = visual.ln_post(x[:, 0, :])
        if visual.proj is not None:
            x = x @ visual.proj

        return x
