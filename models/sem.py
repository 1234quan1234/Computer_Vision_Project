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
    """Interpolate CLIP's fixed positional embeddings to a new spatial resolution.

    CLIP ViT-B/16 was pre-trained on 224×224 images with 14×14 = 196 patches.
    When using larger input images (e.g. 320×320 -> 20×20 = 400 patches),
    the positional embeddings must be spatially interpolated to match.
    The CLS token embedding is preserved unchanged.

    Args:
        clip_model: A loaded CLIP model instance.
        img_size: Target input image size (both height and width).
        patch_size: Patch size of the ViT (16 for ViT-B/16).

    Returns:
        Interpolated positional embeddings of shape
        ``(1, 1 + new_size*new_size, embed_dim)``.
    """
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
    """CLIP ViT-B/16 Semantic Encoder Module (SEM) for vehicle ReID.

    Wraps the visual encoder of CLIP ViT-B/16 to extract 512-dim semantic
    features from vehicle images at arbitrary resolutions (via positional
    embedding interpolation). Applies a **partial fine-tuning** strategy:

      - **Frozen**: Patch embedding (conv1), most transformer blocks, and
        class embedding are kept frozen to preserve CLIP's pre-trained
        visual-semantic knowledge.
      - **Unfrozen**: The last N transformer blocks, LayerNorm (ln_post),
        and projection matrix (proj) are fine-tuned to adapt CLIP's
        features to the vehicle domain.

    This strategy balances between retaining CLIP's strong generalization
    and learning domain-specific vehicle features, while limiting GPU
    memory usage (only a fraction of parameters receive gradients).

    Args:
        model_name: CLIP model variant to load (default: ``'ViT-B/16'``).
        img_size: Input image resolution (must match training transforms).
        patch_size: ViT patch size (16 for ViT-B/16).
        unfreeze_last_blocks: Number of final transformer blocks to unfreeze.
            Set to 0 to fully freeze the visual encoder.
    """

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

        for param in self.visual.ln_post.parameters():
            param.requires_grad = True
        if self.visual.proj is not None:
            self.visual.proj.requires_grad = True

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract CLIP semantic features from a batch of images.

        Manually runs the ViT forward pass (patch embed → CLS token →
        positional encoding → transformer → LN → projection) to support
        interpolated positional embeddings at non-standard resolutions.

        Args:
            images: CLIP-normalized images of shape ``(B, 3, img_size, img_size)``.

        Returns:
            Semantic feature vectors of shape ``(B, 512)``.
        """
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
