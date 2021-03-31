import torch
import torch.nn as nn


import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out + x

class ConvBlock(nn.Module):
    """ implement conv+ReLU two times """
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        conv_relu = []
        conv_relu.append(nn.Conv2d(in_channels=in_channels, out_channels=middle_channels,
                                   kernel_size=3, padding=1, stride=1))
        conv_relu.append(nn.ReLU())
        conv_relu.append(nn.Conv2d(in_channels=middle_channels, out_channels=out_channels,
                                   kernel_size=3, padding=1, stride=1))
        conv_relu.append(nn.ReLU())
        self.conv_ReLU = nn.Sequential(*conv_relu)
    def forward(self, x):
        out = self.conv_ReLU(x)
        return out

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)
        self.Relu = nn.ReLU()
    def forward(self, x):
        x = self.conv1(x)
        x = self.Relu(self.pointwise(x))
        return x

class U_Net(nn.Module):
    def __init__(self):
        super().__init__()
        convs = [] 

        convs.append(ConvBlock(in_channels=3, middle_channels=32, out_channels=32))
        convs.append( nn.MaxPool2d(kernel_size=2, stride=2))
        convs.append(ConvBlock(in_channels=32, middle_channels=64, out_channels=64))
        convs.append( nn.MaxPool2d(kernel_size=2, stride=2))
        convs.append(ConvBlock(in_channels=64, middle_channels=128, out_channels=128))
        convs.append( nn.MaxPool2d(kernel_size=2, stride=2))

        convs.append(SeparableConv2d(in_channels=128, out_channels=128))
        convs.append(SeparableConv2d(in_channels=128, out_channels=128))
        convs.append(SeparableConv2d(in_channels=128, out_channels=128))

        self.features = nn.Sequential(*convs)
        self.last_channel = 128

    def forward(self, x):
        self.features(x)
        return ou

class U_Net_Att(nn.Module):
    def __init__(self):
        super(U_Net_Att, self).__init__()
        convs = [] 
        convs.append(ConvBlock(in_channels=4, middle_channels=32, out_channels=32))
        # convs.append(CBAM(32))
        convs.append( nn.MaxPool2d(kernel_size=2, stride=2))
        convs.append(ConvBlock(in_channels=32, middle_channels=64, out_channels=64))
        convs.append(CBAM(64))
        convs.append( nn.MaxPool2d(kernel_size=2, stride=2))
        convs.append(ConvBlock(in_channels=64, middle_channels=128, out_channels=128))
        convs.append(CBAM(128))
        convs.append( nn.MaxPool2d(kernel_size=2, stride=2))

        convs.append(SeparableConv2d(in_channels=128, out_channels=128))
        convs.append(CBAM(128))
        convs.append(SeparableConv2d(in_channels=128, out_channels=128))
        # convs.append(SeparableConv2d(in_channels=128, out_channels=128))

        self.features = nn.Sequential(*convs)
        self.last_channel = 128


    def forward(self, x):

        out = self.features(x)
        return out