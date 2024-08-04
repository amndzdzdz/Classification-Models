import torch
import torch.nn as nn

class VGG11(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VGG11, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv3 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv4 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv5 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv6 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv7 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv8 = nn.Conv2d(512, 512, 3, 1, 1)
        
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)

        x = self.relu(self.conv2(x))
        x = self.pool(x)

        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool(x)

        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.pool(x)

        x = self.relu(self.conv7(x))
        x = self.relu(self.conv8(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)     

        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)  

        return x


class VGG13(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VGG13, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv6 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv7 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv8 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv9 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv10 = nn.Conv2d(512, 512, 3, 1, 1)

        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)

        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool(x)

        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.pool(x)

        x = self.relu(self.conv7(x))
        x = self.relu(self.conv8(x))
        x = self.pool(x)

        x = self.relu(self.conv9(x))
        x = self.relu(self.conv10(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)     

        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)  

        return x


class VGG16(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VGG16, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv6 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv7 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv8 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv9 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv10 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv11 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv12 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv13 = nn.Conv2d(512, 512, 3, 1, 1)

        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)

        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool(x)

        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x))
        x = self.pool(x)

        x = self.relu(self.conv8(x))
        x = self.relu(self.conv9(x))
        x = self.relu(self.conv10(x))
        x = self.pool(x)

        x = self.relu(self.conv11(x))
        x = self.relu(self.conv12(x))
        x = self.relu(self.conv13(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)  

        return x


class VGG19(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VGG16, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv6 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv7 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv8 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv9 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv10 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv11 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv12 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv13 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv14 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv15 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv16 = nn.Conv2d(512, 512, 3, 1, 1)

        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)

        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool(x)

        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x))
        x = self.relu(self.conv8(x))
        x = self.pool(x)

        x = self.relu(self.conv9(x))
        x = self.relu(self.conv10(x))
        x = self.relu(self.conv11(x))
        x = self.relu(self.conv12(x))
        x = self.pool(x)

        x = self.relu(self.conv13(x))
        x = self.relu(self.conv14(x))
        x = self.relu(self.conv15(x))
        x = self.relu(self.conv16(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)  

        return x