import torch
import torch.nn as nn
import torch.nn.functional as F


class DEM(nn.Module):
    def __init__(self):
        super(DEM, self).__init__()

        self.inc_1 = DoubleConv(3, 16)
        self.down1_1 = Down(16, 32)
        self.down2_1 = Down(32, 64)
        self.down3_1 = Down(64, 128)

        self.inc_2 = DoubleConv(6, 32)
        self.down1_2 = Down(32, 64)
        self.down2_2 = Down(64, 128)
        self.down3_2 = Down(128, 256)

        factor = 2
        # factor = 1
        self.down4_1 = Down(128, 256 // factor)
        self.down4_2 = Down(256, 512 // factor)

        self.up1 = Up(256 * 4, 256 // factor)
        self.up2 = Up(128 * 3, 128 // factor)
        self.up3 = Up(64 * 3, 64 // factor)
        self.up4 = Up(32 * 3, 32)
        self.outc = OutConv(32, 1)

    def forward(self, input):
        img1 = input[:, 0:3, :, :]
        img2 = input[:, 3:, :, :]

        x1_1 = self.inc_1(img1)
        x2_1 = self.down1_1(x1_1)
        x3_1 = self.down2_1(x2_1)
        x4_1 = self.down3_1(x3_1)
        x5_1 = self.down4_1(x4_1)
        x1_2 = self.inc_1(img2)
        x2_2 = self.down1_1(x1_2)
        x3_2 = self.down2_1(x2_2)
        x4_2 = self.down3_1(x3_2)
        x5_2 = self.down4_1(x4_2)

        j1 = self.inc_2(input)
        j2 = self.down1_2(j1)
        j3 = self.down2_2(j2)
        j4 = self.down3_2(j3)
        j5 = self.down4_2(j4)

        x = self.up1(torch.cat([x5_1, x5_2, j5], dim=1), torch.cat([x4_1, x4_2, j4], dim=1))
        x = self.up2(x, torch.cat([x3_1, x3_2, j3], dim=1))
        x = self.up3(x, torch.cat([x2_1, x2_2, j2], dim=1))
        x = self.up4(x, torch.cat([x1_1, x1_2, j1], dim=1))
        out = self.outc(x)

        out = F.tanh(out)
        out = out / 2 + 0.5

        return out


class CEM(nn.Module):
    def __init__(self):
        super(CEM, self).__init__()
        self.inc = DoubleConv(7, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2
        # factor = 1
        self.down4 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor)
        self.up2 = Up(256, 128 // factor)
        self.up3 = Up(128, 64 // factor)
        self.up4 = Up(64, 32)
        self.outc = OutConv(32, 2)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)

        out = F.tanh(out)
        out = out / 2 + 0.5

        return out


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)