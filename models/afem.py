import math
import torch
import torch.nn as nn


class GroupedLinear(nn.Module):
    """Grouped linear projection for channel-wise feature transformation.

    Splits the input vector into ``num_groups`` non-overlapping groups and
    applies an independent linear transformation to each group. This is
    more parameter-efficient than a full linear layer and allows the model
    to process different semantic attribute groups independently.

    For a 512-dim input with 32 groups, each group has 16 dimensions and
    its own learnable 16×16 weight matrix, totaling 32×16×16 = 8192
    parameters (vs. 512×512 = 262144 for a full linear layer).

    Args:
        in_channels: Input feature dimension (must be divisible by num_groups).
        num_groups: Number of independent groups (G).
    """

    def __init__(self, in_channels: int = 512, num_groups: int = 32) -> None:
        super().__init__()
        if in_channels % num_groups != 0:
            raise ValueError("in_channels must be divisible by num_groups")

        self.in_channels = in_channels
        self.num_groups = num_groups
        self.group_dim = in_channels // num_groups

        weight = torch.empty(num_groups, self.group_dim, self.group_dim)
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.zeros(num_groups, self.group_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply per-group linear transformation.

        Args:
            x: Input features of shape ``(B, in_channels)``.

        Returns:
            Transformed features of shape ``(B, in_channels)``.
        """
        b, c = x.shape
        x = x.view(b, self.num_groups, self.group_dim)
        out = torch.einsum("bgd,gdh->bgh", x, self.weight) + self.bias
        return out.reshape(b, c)


class AFEM(nn.Module):
    """Adaptive Fine-grained Enhancement Module (AFEM).

    Refines semantic features from CLIP via a group-wise recalibration
    mechanism. The module:
      1. Applies a grouped linear projection to transform features.
      2. Scales each group by a learned sigmoid gate ``group_scale``.
      3. Normalizes with BatchNorm and adds back to the input (residual).

    The residual design ensures that the enhanced features preserve
    the original CLIP representation while selectively amplifying or
    suppressing specific semantic attribute groups (e.g., color, shape).

    Args:
        in_channels: Feature dimension (default: 512).
        num_groups: Number of semantic groups (G) for recalibration.
    """

    def __init__(self, in_channels: int = 512, num_groups: int = 32) -> None:
        super().__init__()
        self.group_linear = GroupedLinear(in_channels=in_channels, num_groups=num_groups)
        self.group_scale = nn.Parameter(torch.ones(num_groups))
        self.act = nn.GELU()
        self.norm = nn.BatchNorm1d(in_channels)

        self.num_groups = num_groups
        self.group_dim = in_channels // num_groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Enhance semantic features via group-wise recalibration.

        Args:
            x: CLIP semantic features of shape ``(B, in_channels)``.

        Returns:
            Enhanced features of shape ``(B, in_channels)``, computed as
            ``x + BN(GroupedLinear(x) * sigmoid(group_scale))``.
        """
        y = self.group_linear(x)
        y = self.act(y)

        b, c = y.shape
        y = y.view(b, self.num_groups, self.group_dim)
        scale = torch.sigmoid(self.group_scale).view(1, self.num_groups, 1)
        y = y * scale
        y = y.view(b, c)

        y = self.norm(y)
        return x + y
