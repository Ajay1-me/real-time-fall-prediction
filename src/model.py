"""Model architectures for fall detection."""

import torch
import torch.nn as nn


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class FallDetectorCNN(nn.Module):
    def __init__(self, in_channels: int = 9, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 200 -> 100

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 100 -> 50

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # -> (batch, 128, 1)
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)  # (B, 128, 1)
        x = x.squeeze(-1)  # (B, 128)
        return self.classifier(x)  # (B, 2)


class TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, dilation: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        pad = (k - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=pad, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        y = y[..., :x.shape[-1]]  # crop causal padding overshoot to keep length
        y = self.dropout(torch.relu(self.bn(y)))
        return y + self.down(x)


class FallDetectorTCN(nn.Module):
    def __init__(self, in_channels: int = 9, num_classes: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            TCNBlock(in_channels, 32, dilation=1),
            TCNBlock(32, 64, dilation=2),
            TCNBlock(64, 128, dilation=4),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x).squeeze(-1)
        return self.fc(x)
