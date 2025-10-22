from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, p_drop: float=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, padding=1, bias=False)
        self.in1 = nn.InstanceNorm2d(out_channel, affine=True)
        self.act1 = nn.LeakyReLU(0.01, inplace=True)

        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(out_channel, affine=True)
        self.act2 = nn.LeakyReLU(0.01, inplace=True)

        if p_drop > 0:
            self.dropout = nn.Dropout2d(p_drop)
        else:
            self.dropout = nn.Identity()

    def forward(self, x):
        x = self.act1(self.in1(self.conv1(x)))
        x = self.dropout(x)
        x = self.act2(self.in2(self.conv2(x)))
        return x


class DownSample(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, p_drop: float=0.0):
        super().__init__()
        self.down = nn.Conv2d(in_channel, out_channel, 3, padding=1, stride=2, bias=False)
        self.norm = nn.InstanceNorm2d(out_channel, affine=True)
        self.act = nn.LeakyReLU(0.01, inplace=True)
        self.block = ConvBlock(out_channel, out_channel, p_drop)

    def forward(self, x):
        x = self.act(self.norm(self.down(x)))
        x = self.block(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, p_drop: float=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channel // 2, in_channel // 2, kernel_size=2, stride=2)
        self.block = ConvBlock(in_channel, out_channel, p_drop)

    def forward(self, x, skip):
        x = self.up(x)

        if (x.shape[-2], x.shape[-1]) != (skip.shape[-2], skip.shape[-1]):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        x = self.block(x)
        return x


class IUNet2D(nn.Module):
    def __init__(self, in_channels:int=1, out_channels:int=1, base_channel:int=32, depth:int=4, p_drop:float=0.0, deep_supervision:bool=False):
        super().__init__()
        self.deep_supervision = deep_supervision

        channels = []
        for i in range(depth):
            channels.append(base_channel * (2**i))

        self.encoder0 = ConvBlock(in_channels, channels[0], p_drop)
        self.downs = nn.ModuleList()
        for i in range(1, depth):
            self.downs.append(DownSample(channels[i-1], channels[i], p_drop))

        self.bottleneck = ConvBlock(channels[-1], channels[-1], p_drop)

        self.ups = nn.ModuleList()
        dec_channels = list(reversed(channels))
        self.dec_blocks = nn.ModuleList()
        for i in range(depth - 1):
            up_output = dec_channels[i + 1]
            self.ups.append(nn.ConvTranspose2d(dec_channels[i], up_output, kernel_size=2, stride=2))
            input_block = up_output * 2
            self.dec_blocks.append(ConvBlock(input_block, up_output, p_drop))

        self.head = nn.Conv2d(dec_channels[-1], out_channels, kernel_size=1)
        if deep_supervision:
            self.aux_heads = nn.ModuleList()
            for i in range(depth - 2):
                self.aux_heads.append(nn.Conv2d(dec_channels[i+1], out_channels, kernel_size=1))
        else:
            self.aux_heads = None

        kaiming_initialization(self)

    def forward(self, x):
        skips = []
        x0 = self.encoder0(x)
        skips.append(x0)
        x = x0
        for d in self.downs:
            x = d(x)
            skips.append(x)

        x = self.bottleneck(x)

        aux_log = []
        for i in range(len(self.ups)):
            up = self.ups[i]
            skip = skips[-(i+2)]
            if up.shape[-2:] != skip[-2:]:
                up = F.interpolate(up, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([skip, up], dim=1)
            x = self.dec_blocks[i](x)

            if self.deep_supervision and i < len(self.ups) - 1:
                aux_log.append(self.aux_heads[i](x))

        out = self.head(x)
        if self.deep_supervision:
            final_hw = out.shape[-2:]
            aux_log = []
            for a in aux_log:
                aux_log.append(F.interpolate(a, size=final_hw, mode="bilinear", align_corners=False))
            return [out] + aux_log
        return out


def kaiming_initialization(module: nn.Module):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, a=0.01)  # LeakyReLU a=0.01
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)