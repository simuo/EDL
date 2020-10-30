import torch
import torch.nn as nn


class DepthWiseNet(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(DepthWiseNet, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(inchannels, inchannels, kernel_size=3, stride=1, groups=inchannels),
            nn.Conv2d(inchannels, outchannels, kernel_size=1, stride=1)
        )

    def forward(self, x):
        return self.layer(x)


if __name__ == '__main__':
    data = torch.randn(1, 3, 100, 100)
    net = DepthWiseNet()
    print('parameters:', sum(param.numel() for param in net.layer[0].parameters()))
    print('parameters:', sum(param.numel() for param in net.layer[1].parameters()))
    output = net(data)
    print(output.shape)
    for name, param in net.named_parameters():
        print(name, param.shape)
