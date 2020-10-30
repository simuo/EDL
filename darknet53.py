import torch
import torch.nn as nn
import torch.nn.functional as F
from cfg import *


class UpsampleLayer(nn.Module):
    def __init__(self):
        super(UpsampleLayer, self).__init__()

    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode='nearest')


class ConvolutionalLayer(nn.Module):
    def __init__(self, inchannels, outputchannels, kerner_size, stride, padding, bias=False):
        super(ConvolutionalLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=inchannels, out_channels=outputchannels, kernel_size=kerner_size, stride=stride,
                      padding=padding, bias=bias),
            nn.BatchNorm2d(outputchannels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.layer(x)


class ConvolutionalSets(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(ConvolutionalSets, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(inchannels, outchannels, 1, 1, 0),
            nn.Conv2d(outchannels, inchannels, 3, 1, 1),
            nn.Conv2d(inchannels, outchannels, 1, 1, 0),
            nn.Conv2d(outchannels, inchannels, 3, 1, 1),
            nn.Conv2d(inchannels, outchannels, 1, 1, 0),
        )

    def forward(self, x):
        return self.layer(x)


class DownSampleLayer(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(DownSampleLayer, self).__init__()
        self.layer = nn.Sequential(
            ConvolutionalLayer(inchannels, outchannels, 3, 2, 1)
        )

    def forward(self, x):
        return self.layer(x)


class Residualayer(nn.Module):
    def __init__(self, inchannels):
        super(Residualayer, self).__init__()
        self.layer = nn.Sequential(
            ConvolutionalLayer(inchannels, inchannels // 2, 1, 1, 0),
            ConvolutionalLayer(inchannels // 2, inchannels, 3, 1, 1)
        )

    def forward(self, x):
        return x + self.layer(x)


class MainNet(nn.Module):
    def __init__(self,cls):
        super(MainNet, self).__init__()
        self.trunk_52 = nn.Sequential(
            ConvolutionalLayer(3, 32, 3, 1, 1),  # 416
            ConvolutionalLayer(32, 64, 3, 2, 1),  # 208
            Residualayer(64),  # 208
            DownSampleLayer(64, 128),  # 104
            Residualayer(128),
            Residualayer(128),  # 104
            DownSampleLayer(128, 256),  # 52
            Residualayer(256),
            Residualayer(256),
            Residualayer(256),
            Residualayer(256),
            Residualayer(256),
            Residualayer(256),
            Residualayer(256),
            Residualayer(256)  # 52
        )

        self.trunk_26 = nn.Sequential(
            DownSampleLayer(256, 512),  # 26
            Residualayer(512),
            Residualayer(512),
            Residualayer(512),
            Residualayer(512),
            Residualayer(512),
            Residualayer(512),
            Residualayer(512),
            Residualayer(512)  # 26
        )
        self.trunk_13 = nn.Sequential(
            DownSampleLayer(512, 1024),  # 13
            Residualayer(1024),
            Residualayer(1024),
            Residualayer(1024),
            Residualayer(1024)  # 13
        )

        self.convset_13 = nn.Sequential(
            ConvolutionalSets(1024, 512)
        )
        self.detection_13 = nn.Sequential(
            ConvolutionalLayer(512, 1024, 3, 1, 1),
            nn.Conv2d(1024, 3 * (5 + CLASS_NUM), 1, 1)
        )
        self.up_26 = nn.Sequential(
            ConvolutionalLayer(512, 1024, 1, 1, 0),
            UpsampleLayer()
        )
        self.convset_26 = torch.nn.Sequential(
            ConvolutionalLayer(1536, 256, 1, 1, 0),
            ConvolutionalSets(256, 512)
        )

        self.detection_26 = torch.nn.Sequential(
            ConvolutionalLayer(512, 512, 3, 1, 1),
            torch.nn.Conv2d(512, 3 * (5 + CLASS_NUM), 1, 1, 0)
        )

        self.up_52 = torch.nn.Sequential(
            ConvolutionalLayer(512, 128, 1, 1, 0),  #
            UpsampleLayer()
        )

        self.convset_52 = torch.nn.Sequential(
            ConvolutionalLayer(384, 128, 1, 1, 0),
            ConvolutionalSets(128, 256)
        )

        self.detection_52 = torch.nn.Sequential(
            ConvolutionalLayer(256, 256, 3, 1, 1),
            torch.nn.Conv2d(256, 3 * (5 + CLASS_NUM), 1, 1, 0)
        )

    def forward(self, x):
        h_52 = self.trunk_52(x)
        h_26 = self.trunk_26(h_52)
        h_13 = self.trunk_13(h_26)

        convset_out_13 = self.convset_13(h_13)
        detetion_out_13 = self.detection_13(convset_out_13)

        up_out_26 = self.up_26(convset_out_13)
        route_out_26 = torch.cat((up_out_26, h_26), dim=1)
        convset_out_26 = self.convset_26(route_out_26)
        detetion_out_26 = self.detection_26(convset_out_26)

        up_out_52 = self.up_52(convset_out_26)
        route_out_52 = torch.cat((up_out_52, h_52), dim=1)
        convset_out_52 = self.convset_52(route_out_52)
        detetion_out_52 = self.detection_52(convset_out_52)

        return detetion_out_13, detetion_out_26, detetion_out_52


if __name__ == '__main__':
    image = torch.randint(0, 255, (1, 3, 416, 416), dtype=torch.float32)
    net = MainNet(14)
    out, out1, out2 = net(image)
    print(out.shape)
    print(out1.shape)
    print(out2.shape)