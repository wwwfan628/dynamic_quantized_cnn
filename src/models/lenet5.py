import torch
from torch import nn
from utils.ste_layers import LinearQuantized, Conv2dQuantized
from utils.masked_layers import LinearMasked, Conv2dMasked

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class LeNet5(nn.Module):
    def __init__(self, input_channel=1, n_classes=10):
        super(LeNet5, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(input_channel, 20, 5, 1), nn.MaxPool2d(2, 2), nn.ReLU(inplace=False),
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


class LeNet5_Masked(nn.Module):
    def __init__(self, input_channel=1, n_classes=10):
        super(LeNet5_Masked, self).__init__()
        self.features = nn.Sequential(Conv2dMasked(input_channel, 20, 5, 1), nn.MaxPool2d(2, 2), nn.ReLU(inplace=False),
                                      Conv2dMasked(20, 50, 5, 1), nn.MaxPool2d(2, 2), nn.ReLU(inplace=False))
        self.classifier = nn.Sequential(LinearMasked(4 * 4 * 50, 500), nn.ReLU(inplace=False), LinearMasked(500, n_classes))
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 4 * 4 * 50)
        x = self.classifier(x)
        return x


class LeNet5_Quantized(nn.Module):
    def __init__(self, input_channel=1, n_classes=10, num_quantized_weight_values=16):
        super(LeNet5_Quantized, self).__init__()
        self.quantized_weight_values = torch.zeros(num_quantized_weight_values).to(device)
        self.features = nn.Sequential(
            Conv2dQuantized(input_channel, 20, 5, quantized_weight_values=self.quantized_weight_values, stride=1),
            nn.MaxPool2d(2, 2), nn.ReLU(inplace=False),
            Conv2dQuantized(20, 50, 5, quantized_weight_values=self.quantized_weight_values, stride=1),
            nn.MaxPool2d(2, 2), nn.ReLU(inplace=False))
        self.classifier = nn.Sequential(
            LinearQuantized(4 * 4 * 50, 500, quantized_weight_values=self.quantized_weight_values),
            nn.ReLU(inplace=False),
            LinearQuantized(500, n_classes, quantized_weight_values=self.quantized_weight_values))
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 4 * 4 * 50)
        x = self.classifier(x)
        return x