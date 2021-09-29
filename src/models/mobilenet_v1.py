import torch.nn as nn
from utils.ste_layers import LinearQuantized, Conv2dQuantized
from utils.masked_layers import LinearMasked, Conv2dMasked

class MobileNetV1(nn.Module):
    def __init__(self, input_channel, n_classes):
        super(MobileNetV1, self).__init__()

        def conv_bn(ch_in, ch_out, stride):
            return nn.Sequential(
                nn.Conv2d(ch_in, ch_out, 3, stride, 1, bias=False),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True)
                )

        def conv_dw(ch_in, ch_out, stride):
            return nn.Sequential(
                # depthwise
                nn.Conv2d(ch_in, ch_in, 3, stride, 1, groups=ch_in, bias=False),
                nn.BatchNorm2d(ch_in),
                nn.ReLU(inplace=True),

                # pointwise
                nn.Conv2d(ch_in, ch_out, 1, 1, 0, bias=False),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True),
                )

        self.model = nn.Sequential(
            conv_bn(input_channel, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(1024, n_classes)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x



class MobileNetV1_Masked(nn.Module):
    def __init__(self, input_channel, n_classes):
        super(MobileNetV1_Masked, self).__init__()

        def conv_bn(ch_in, ch_out, stride):
            return nn.Sequential(
                Conv2dMasked(ch_in, ch_out, 3, stride, 1, bias=False),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True)
                )

        def conv_dw(ch_in, ch_out, stride):
            return nn.Sequential(
                # depthwise
                Conv2dMasked(ch_in, ch_in, 3, stride, 1, groups=ch_in, bias=False),
                nn.BatchNorm2d(ch_in),
                nn.ReLU(inplace=True),

                # pointwise
                Conv2dMasked(ch_in, ch_out, 1, 1, 0, bias=False),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True),
                )

        self.model = nn.Sequential(
            conv_bn(input_channel, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = LinearMasked(1024, n_classes)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x



class MobileNetV1_Quantized(nn.Module):
    def __init__(self, input_channel, n_classes):
        super(MobileNetV1_Quantized, self).__init__()

        def conv_bn(ch_in, ch_out, stride):
            return nn.Sequential(
                Conv2dQuantized(ch_in, ch_out, 3, stride, 1, bias=False),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True)
                )

        def conv_dw(ch_in, ch_out, stride):
            return nn.Sequential(
                # depthwise
                Conv2dQuantized(ch_in, ch_in, 3, stride, 1, groups=ch_in, bias=False),
                nn.BatchNorm2d(ch_in),
                nn.ReLU(inplace=True),

                # pointwise
                Conv2dQuantized(ch_in, ch_out, 1, 1, 0, bias=False),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True),
                )

        self.model = nn.Sequential(
            conv_bn(input_channel, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = LinearQuantized(1024, n_classes)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x
