from __future__ import annotations
import torch
import torch.nn as nn

class CustomNet(nn.Module):
    def __init__(self, num_classes: int = 200):
        super(CustomNet, self).__init__()
        self.conv1 = nn.Conv2d(3,   64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.avg   = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1   = nn.Linear(256, num_classes)   # logits for CrossEntropy

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect images resized to 224x224 in the dataloader transforms
        x = self.pool(torch.relu(self.conv1(x)))  # -> 112x112
        x = self.pool(torch.relu(self.conv2(x)))  # -> 56x56
        x = self.pool(torch.relu(self.conv3(x)))  # -> 28x28
        x = self.avg(x)                           # -> B x 256 x 1 x 1
        x = torch.flatten(x, 1)                   # -> B x 256
        return self.fc1(x)                        # -> logits