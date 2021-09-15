from torch import nn


class LeNet5(nn.Module):
    def __init__(self, in_channels=1, n_classes=10):
        super(LeNet5, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(in_channels, 20, 5, 1), nn.MaxPool2d(2, 2), nn.ReLU(inplace=False),
                                      nn.Conv2d(20, 50, 5, 1), nn.MaxPool2d(2, 2), nn.ReLU(inplace=False))
        self.classifier = nn.Sequential(nn.Linear(4 * 4 * 50, 500), nn.ReLU(inplace=False), nn.Linear(500, n_classes))
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 4 * 4 * 50)
        x = self.classifier(x)
        return x