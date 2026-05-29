import math
import torch
import torch.nn as nn


class GroupedLinear(nn.Module):
    """Grouped linear projection for channel-wise enhancement."""

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
        b, c = x.shape
        x = x.view(b, self.num_groups, self.group_dim)
        out = torch.einsum("bgd,gdh->bgh", x, self.weight) + self.bias
        return out.reshape(b, c)


class AFEM(nn.Module):
    """Adaptive Fine-grained Enhancement Module."""

    def __init__(self, in_channels: int = 512, num_groups: int = 32) -> None:
        super().__init__()
        self.group_linear = GroupedLinear(in_channels=in_channels, num_groups=num_groups)
        self.group_scale = nn.Parameter(torch.ones(num_groups))
        self.act = nn.GELU()
        self.norm = nn.BatchNorm1d(in_channels)

        self.num_groups = num_groups
        self.group_dim = in_channels // num_groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.group_linear(x)
        y = self.act(y)

        b, c = y.shape
        y = y.view(b, self.num_groups, self.group_dim)
        scale = torch.sigmoid(self.group_scale).view(1, self.num_groups, 1)
        y = y * scale
        y = y.view(b, c)

        y = self.norm(y)
        return x + y
