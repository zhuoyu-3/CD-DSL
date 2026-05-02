import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .config import CDDSLConfig


class MNISTNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def resolve_group_count(num_channels: int, requested_groups: int) -> int:
    groups = max(1, min(int(requested_groups), int(num_channels)))
    while num_channels % groups != 0 and groups > 1:
        groups -= 1
    return groups


def build_norm2d(num_channels: int, norm_kind: str, group_norm_groups: int) -> nn.Module:
    norm_kind = norm_kind.lower()
    if norm_kind == "batch":
        return nn.BatchNorm2d(num_channels)
    if norm_kind == "group":
        return nn.GroupNorm(resolve_group_count(num_channels, group_norm_groups), num_channels)
    raise ValueError(f"Unsupported norm kind: {norm_kind}")


def build_model(cfg: CDDSLConfig) -> nn.Module:
    if cfg.dataset.upper() == "MNIST":
        return MNISTNet()
    if cfg.norm_kind.lower() == "group":
        net = models.resnet18(
            weights=None,
            norm_layer=lambda channels: nn.GroupNorm(
                resolve_group_count(channels, cfg.group_norm_groups),
                channels,
            ),
        )
    else:
        net = models.resnet18(weights=None)
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    net.fc = nn.Linear(net.fc.in_features, 10)
    return net
