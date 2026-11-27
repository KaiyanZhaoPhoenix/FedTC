"""Three-stage CIFAR ResNet-20: 3x3 stem, 3 basic blocks per stage."""
from torch import nn


class Block(nn.Module):
    def __init__(self, inputs, width, stride):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(inputs, width, 3, stride, 1, bias=False),
                                  nn.BatchNorm2d(width), nn.ReLU(),
                                  nn.Conv2d(width, width, 3, 1, 1, bias=False), nn.BatchNorm2d(width))
        self.shortcut = nn.Identity() if stride == 1 else nn.Sequential(
            nn.Conv2d(inputs, width, 1, stride, bias=False), nn.BatchNorm2d(width))

    def forward(self, x):
        return (self.body(x) + self.shortcut(x)).relu()


def resnet20(num_classes=10):
    layers = [nn.Conv2d(3, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU()]
    inputs = 16
    for width in (16, 32, 64):
        for _ in range(3):
            layers.append(Block(inputs, width, 1 if inputs == width else 2))
            inputs = width
    return nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, num_classes))
