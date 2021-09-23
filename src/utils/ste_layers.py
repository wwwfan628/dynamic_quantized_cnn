import torch
import torch.nn as nn
import torch.nn.functional as F

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class ste_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, quantized_weight_values):
        idx = torch.argmin(
            (weight.view(-1).unsqueeze(1).expand(-1, quantized_weight_values.shape[0]) - quantized_weight_values).abs(),
            dim=1).unsqueeze(0)
        quantized_weight = (quantized_weight_values.unsqueeze(0).expand(weight.view(-1).shape[0], -1)).gather(-1, idx).view(weight.shape)
        return quantized_weight

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class LinearQuantized(nn.Linear):
    def __init__(self, in_features, out_features, quantized_weight_values, bias=True):
        super(LinearQuantized, self).__init__(in_features, out_features, bias)
        self.mask_flag = False
        self.mask = torch.zeros(0)
        self.quantized_weight_values = quantized_weight_values.clone().detach().to(device)

    def set_quantized_weight_values(self, quantized_weight_values):
        self.quantized_weight_values = quantized_weight_values.clone().detach().to(device)

    def set_mask(self, mask):
        self.mask = mask
        self.mask_flag = True

    def get_quantized_weight_values(self):
        return self.quantized_weight_values

    def get_mask(self):
        print(self.mask_flag)
        return self.mask

    def forward(self, x):
        weight = ste_function.apply(self.weight, self.quantized_weight_values.clone().detach().to(device))
        if self.mask_flag:
            weight = weight * self.mask
        return F.linear(x, weight, self.bias)


class Conv2dQuantized(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, quantized_weight_values, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dQuantized, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_flag = False
        self.mask = torch.zeros(0)
        self.quantized_weight_values = quantized_weight_values.clone().detach().to(device)

    def set_quantized_weight_values(self, quantized_weight_values):
        self.quantized_weight_values = quantized_weight_values.clone().detach().to(device)

    def set_mask(self, mask):
        self.mask = mask
        self.mask_flag = True

    def get_quantized_weight_values(self):
        return self.quantized_weight_values

    def get_mask(self):
        print(self.mask_flag)
        return self.mask

    def forward(self, x):
        weight = ste_function.apply(self.weight, self.quantized_weight_values.clone().detach().to(device))
        if self.mask_flag:
            weight = weight * self.mask
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
