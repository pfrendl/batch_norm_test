import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ReLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        gain = math.sqrt(2)
        return gain * x.relu()


class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> None:
        super().__init__()
        gain = 1 / math.sqrt(in_channels * kernel_size * kernel_size)
        self.weight = nn.Parameter(
            gain * torch.randn((out_channels, in_channels, kernel_size, kernel_size))
        )
        self.bias = nn.Parameter(torch.zeros((out_channels,)))
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        return F.conv2d(
            input=x,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )


class ConvT2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> None:
        super().__init__()
        gain = 1 / math.sqrt(in_channels * kernel_size * kernel_size)
        self.weight = nn.Parameter(
            gain * torch.randn((in_channels, out_channels, kernel_size, kernel_size))
        )
        self.bias = nn.Parameter(torch.zeros((out_channels,)))
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        return F.conv_transpose2d(
            input=x,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )
