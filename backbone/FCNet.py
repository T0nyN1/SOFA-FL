import torch.nn as nn

class FCNet(nn.Module):
    """
    A Fully Connected Neural Network for testing.
    """
    def __init__(self, in_dim=784, out_dim=10):
        super(FCNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x