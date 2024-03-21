import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
nonlinearity = partial(F.relu, inplace=True)

def search_threshold(weight):
    bn_min=torch.min(weight)
    bn_max=torch.max(weight)
    bin = len(weight)
    # bin=100
    hist_y = torch.histc(input=weight, bins=bin)
    hist_y_diff = torch.diff(hist_y)
    for i in range(len(hist_y_diff) - 1):
        if hist_y_diff[i] <= 0 < hist_y_diff[i + 1]:
            threshold = bn_min+(i+2)*(bn_max-bn_min)/bin
            return threshold


class CMIP(nn.Module):
    def __init__(self):
        super(CMIP, self).__init__()

    def forward(self, x, bn):
        bn1, bn2 = bn[0].weight.abs(), bn[1].weight.abs()
        bn_threshold1 = search_threshold(bn1)
        bn_threshold2 = search_threshold(bn2)
        x1, x2 = torch.zeros_like(x[0]), torch.zeros_like(x[1])
        x1[:, bn1 >= bn_threshold1] = x[0][:, bn1 >= bn_threshold1]
        x1[:, bn1 < bn_threshold1] = x[1][:, bn1 < bn_threshold1]
        x2[:, bn2 >= bn_threshold2] = x[1][:, bn2 >= bn_threshold2]
        x2[:, bn2 < bn_threshold2] = x[0][:, bn2 < bn_threshold2]
        return x1,x2
class ModuleParallel(nn.Module):
    def __init__(self, module):
        super(ModuleParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel):
        return [self.module(x) for x in x_parallel]


class BatchNorm2dParallel(nn.Module):
    def __init__(self, num_features, num_parallel=2):
        super(BatchNorm2dParallel, self).__init__()
        for i in range(int(num_parallel)):

            setattr(self, 'bn_' + str(i), nn.BatchNorm2d(num_features))

    def forward(self, x_parallel):
        return [getattr(self, 'bn_' + str(i))(x) for i, x in enumerate(x_parallel)]
def conv3x3(in_planes, out_planes, stride=1, bias=False):
    "3x3 convolution with padding"
    return ModuleParallel(nn.Conv2d(in_planes, out_planes, kernel_size=3,
                                    stride=stride, padding=1, bias=bias))


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return ModuleParallel(nn.Conv2d(in_planes, out_planes, kernel_size=1,
                                    stride=stride, padding=0, bias=bias))


class DBlock_parallel(nn.Module):
    def __init__(self, channel,num_parallel):
        super(DBlock_parallel, self).__init__()
        self.dilate1 = ModuleParallel(nn.Conv2d(
            channel, channel, kernel_size=3, dilation=1, padding=1))
        self.dilate2 = ModuleParallel(nn.Conv2d(
            channel, channel, kernel_size=3, dilation=2, padding=2))
        self.dilate3 = ModuleParallel(nn.Conv2d(
            channel, channel, kernel_size=3, dilation=4, padding=4))
        self.dilate4 = ModuleParallel(nn.Conv2d(
            channel, channel, kernel_size=3, dilation=8, padding=8))
        self.relu = ModuleParallel(nn.ReLU(inplace=True))
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
        self.num_parallel=num_parallel
    def forward(self, x):
        dilate1_out = self.relu(self.dilate1(x))
        dilate2_out = self.relu(self.dilate2(dilate1_out))
        dilate3_out = self.relu(self.dilate3(dilate2_out))
        dilate4_out = self.relu(self.dilate4(dilate3_out))
        out = [x[l] + dilate1_out[l] + dilate2_out[l] + dilate3_out[l] + dilate4_out[l] for l in range(self.num_parallel)]

        return out
class DecoderBlock_parallel(nn.Module):
    def __init__(self, in_channels, n_filters,num_parallel):
        super(DecoderBlock_parallel, self).__init__()

        self.conv1 = conv1x1(in_channels, in_channels // 4, 1)
        self.norm1 = BatchNorm2dParallel(in_channels // 4, num_parallel)
        self.relu1 =  ModuleParallel(nn.ReLU(inplace=True))
        self.deconv2 = ModuleParallel(nn.ConvTranspose2d(
            in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1
        ))
        self.norm2 = BatchNorm2dParallel(in_channels // 4, num_parallel)
        self.relu2 = ModuleParallel(nn.ReLU(inplace=True))
        self.conv3 = conv1x1(in_channels // 4, n_filters, 1)
        self.norm3 = BatchNorm2dParallel(n_filters, num_parallel)
        self.relu3 = ModuleParallel(nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x
