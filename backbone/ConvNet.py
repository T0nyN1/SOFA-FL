import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    """
    A Convolutional Neural Network (CNN) for image classification, designed to work with single-channel (grayscale) images.
    """
    def __init__(self, in_dim=1, out_dim=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_dim, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc1 = nn.LazyLinear(600)
        self.fc2 = nn.Linear(600, 120)
        self.fc3 = nn.Linear(120, out_dim)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out