import torch 
import torch.nn as nn
from modules.blocks import ResidualBlock, BottleNeckBlock

class ResNet34(nn.Module):
    def __init__(self, in_channels):
        super(ResNet34, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3)
        self.pool = nn.MaxPool2d(3, 2)
        self.avgpool = nn.AvgPool2d(3, 2)
        self.layer1 = self._make_layer(64, 64, 3, 1)
        self.layer2 = self._make_layer(64, 128, 4, 2)
        self.layer3 = self._make_layer(128, 256, 6, 2)
        self.layer4 = self._make_layer(256, 512, 3, 2)
        self.fc1 = nn.Linear(512*3*3, 10)
        self.relu = nn.ReLU()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))

        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride=stride, downsample=downsample))

        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1, downsample=None))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


class ResNet50(nn.Module): 
    def __init__(self, in_channels, blocks):
        super(ResNet50, self).__init__()
        self.in_channels = 64
        self.blocks = blocks
        self.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(3, 2)
        self.avgpool = nn.AvgPool2d(3, 2)

        self.layer1 = self._make_layer(64, self.blocks[0], 1)
        self.layer2 = self._make_layer(128, self.blocks[1], 2)
        self.layer3 = self._make_layer(256, self.blocks[2], 2)
        self.layer4 = self._make_layer(512, self.blocks[3], 2)
        self.fc1 = nn.Linear(2048*3*3, 10)
        self.relu = nn.ReLU()

    def _make_layer(self, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1) 
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(BottleNeckBlock(self.in_channels, channels, stride))
            self.in_channels = channels * 4

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
