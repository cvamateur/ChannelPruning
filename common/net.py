import json

import torch
import torch.nn as nn


class Concat(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, inputs):
        return torch.cat(inputs, self.dim)


class DWConv(nn.Module):

    def __init__(self, in_channels, kernel_size, stride, padding=1):
        super().__init__()
        self.dw = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pw = nn.Conv2d(in_channels, in_channels, 1, 1)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        return x


class MNIST_Net(nn.Module):

    def __init__(self, factor: float = 1.0):
        super().__init__()

        _h = int(factor * 16)

        self.conv0 = nn.Conv2d(1, _h, 3, 1, 1)
        self.relu0 = nn.LeakyReLU(inplace=True)

        self.conv1 = nn.Conv2d(_h, _h, 4, 2, 1)
        self.relu1 = nn.LeakyReLU(inplace=True)

        self.branch0 = nn.Sequential(
            nn.Conv2d(_h, 2*_h, 4, 2, 1),
            nn.LeakyReLU(inplace=True),
            DWConv(2*_h, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(2*_h, 4 * _h, 4, 2, 1),
            nn.LeakyReLU(inplace=True),
        )

        self.branch1 = nn.Sequential(
            nn.Conv2d(_h, 2 * _h, 4, 2, 1),
            nn.LeakyReLU(inplace=True),
            DWConv(2 * _h, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(2*_h, 4 * _h, 4, 2, 1),
            nn.LeakyReLU(inplace=True),
        )

        self.cat = Concat(dim=1)
        self.conv2 = DWConv(8*_h, 3, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out = nn.Conv2d(8*_h, 10, 1, 1)

    def forward(self, x):
        x = self.conv0(x)
        x = self.relu0(x)
        x = self.conv1(x)
        x = self.relu1(x)

        x0 = self.branch0(x)
        x1 = self.branch1(x)

        x = self.cat([x0, x1])
        x = self.conv2(x)
        x = self.pool(x)
        x = self.out(x)
        return x.flatten(1)




if __name__ == '__main__':
    from torchsummary import summary

    net = MNIST_Net()
    print(net)
    fake_inputs = torch.rand([2, 1, 28, 28])
    summary(net, fake_inputs, device=torch.device("cpu"))

