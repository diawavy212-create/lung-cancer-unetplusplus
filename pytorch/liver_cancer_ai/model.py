import torch
from torch import nn
import torch.nn.functional as F


class ECABlock(nn.Module):
    """Efficient Channel Attention for 2D CT feature maps."""

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        weights = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        weights = self.conv(weights)
        weights = self.activation(weights).transpose(-1, -2).unsqueeze(-1)
        return x * weights.expand_as(x)


class DilatedConvBlock(nn.Module):
    """Convolution block with optional dilation followed by ECA attention."""

    def __init__(self, in_channels, out_channels, dilation=1, use_eca=True):
        super().__init__()
        padding = dilation
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.eca = ECABlock(out_channels) if use_eca else nn.Identity()

    def forward(self, x):
        return self.eca(self.block(x))


class UNet(nn.Module):
    """Compact 2D U-Net baseline for liver tumor segmentation."""

    def __init__(self, in_channels=1, num_classes=2, base_channels=32, use_eca=False, use_dilation=False):
        super().__init__()
        filters = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8, base_channels * 16]
        dilations = [1, 1, 2, 2, 3] if use_dilation else [1, 1, 1, 1, 1]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc0 = DilatedConvBlock(in_channels, filters[0], dilation=dilations[0], use_eca=use_eca)
        self.enc1 = DilatedConvBlock(filters[0], filters[1], dilation=dilations[1], use_eca=use_eca)
        self.enc2 = DilatedConvBlock(filters[1], filters[2], dilation=dilations[2], use_eca=use_eca)
        self.enc3 = DilatedConvBlock(filters[2], filters[3], dilation=dilations[3], use_eca=use_eca)
        self.center = DilatedConvBlock(filters[3], filters[4], dilation=dilations[4], use_eca=use_eca)

        self.dec3 = DilatedConvBlock(filters[4] + filters[3], filters[3], dilation=dilations[3], use_eca=use_eca)
        self.dec2 = DilatedConvBlock(filters[3] + filters[2], filters[2], dilation=dilations[2], use_eca=use_eca)
        self.dec1 = DilatedConvBlock(filters[2] + filters[1], filters[1], dilation=dilations[1], use_eca=use_eca)
        self.dec0 = DilatedConvBlock(filters[1] + filters[0], filters[0], dilation=dilations[0], use_eca=use_eca)
        self.final = nn.Conv2d(filters[0], num_classes, kernel_size=1)

    def up(self, x, target):
        return F.interpolate(x, size=target.shape[2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        x0 = self.enc0(x)
        x1 = self.enc1(self.pool(x0))
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.center(self.pool(x3))

        y3 = self.dec3(torch.cat([x3, self.up(x4, x3)], dim=1))
        y2 = self.dec2(torch.cat([x2, self.up(y3, x2)], dim=1))
        y1 = self.dec1(torch.cat([x1, self.up(y2, x1)], dim=1))
        y0 = self.dec0(torch.cat([x0, self.up(y1, x0)], dim=1))
        return self.final(y0)


class LiverECAUNetPlusPlus(nn.Module):
    """
    Lightweight UNet++ for liver CT slices.

    The network keeps the nested UNet++ skip topology, adds ECA channel attention
    in every convolution block, and uses dilated convolutions in deeper stages to
    enlarge the lesion context without reducing feature-map resolution.
    """

    def __init__(
        self,
        in_channels=1,
        num_classes=2,
        base_channels=32,
        deep_supervision=False,
        use_eca=True,
        use_dilation=True,
    ):
        super().__init__()
        filters = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8, base_channels * 16]
        dilations = [1, 1, 2, 2, 3] if use_dilation else [1, 1, 1, 1, 1]
        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = lambda x, target: F.interpolate(x, size=target.shape[2:], mode="bilinear", align_corners=False)

        self.conv0_0 = DilatedConvBlock(in_channels, filters[0], dilation=dilations[0], use_eca=use_eca)
        self.conv1_0 = DilatedConvBlock(filters[0], filters[1], dilation=dilations[1], use_eca=use_eca)
        self.conv2_0 = DilatedConvBlock(filters[1], filters[2], dilation=dilations[2], use_eca=use_eca)
        self.conv3_0 = DilatedConvBlock(filters[2], filters[3], dilation=dilations[3], use_eca=use_eca)
        self.conv4_0 = DilatedConvBlock(filters[3], filters[4], dilation=dilations[4], use_eca=use_eca)

        self.conv0_1 = DilatedConvBlock(filters[0] + filters[1], filters[0], dilation=dilations[0], use_eca=use_eca)
        self.conv1_1 = DilatedConvBlock(filters[1] + filters[2], filters[1], dilation=dilations[1], use_eca=use_eca)
        self.conv2_1 = DilatedConvBlock(filters[2] + filters[3], filters[2], dilation=dilations[2], use_eca=use_eca)
        self.conv3_1 = DilatedConvBlock(filters[3] + filters[4], filters[3], dilation=dilations[3], use_eca=use_eca)

        self.conv0_2 = DilatedConvBlock(filters[0] * 2 + filters[1], filters[0], dilation=dilations[0], use_eca=use_eca)
        self.conv1_2 = DilatedConvBlock(filters[1] * 2 + filters[2], filters[1], dilation=dilations[1], use_eca=use_eca)
        self.conv2_2 = DilatedConvBlock(filters[2] * 2 + filters[3], filters[2], dilation=dilations[2], use_eca=use_eca)

        self.conv0_3 = DilatedConvBlock(filters[0] * 3 + filters[1], filters[0], dilation=dilations[0], use_eca=use_eca)
        self.conv1_3 = DilatedConvBlock(filters[1] * 3 + filters[2], filters[1], dilation=dilations[1], use_eca=use_eca)

        self.conv0_4 = DilatedConvBlock(filters[0] * 4 + filters[1], filters[0], dilation=dilations[0], use_eca=use_eca)

        self.final1 = nn.Conv2d(filters[0], num_classes, kernel_size=1)
        self.final2 = nn.Conv2d(filters[0], num_classes, kernel_size=1)
        self.final3 = nn.Conv2d(filters[0], num_classes, kernel_size=1)
        self.final4 = nn.Conv2d(filters[0], num_classes, kernel_size=1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0, x0_0)], dim=1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0, x1_0)], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1, x0_0)], dim=1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0, x2_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1, x1_0)], dim=1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2, x0_0)], dim=1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0, x3_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1, x2_0)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2, x1_0)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3, x0_0)], dim=1))

        if self.deep_supervision:
            return [self.final1(x0_1), self.final2(x0_2), self.final3(x0_3), self.final4(x0_4)]

        return self.final4(x0_4)


def build_liver_eca_unetpp(in_channels=1, num_classes=2, base_channels=32, deep_supervision=False):
    return LiverECAUNetPlusPlus(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        deep_supervision=deep_supervision,
    )


def build_segmentation_model(
    model_name="eca-dilated-unetpp",
    in_channels=1,
    num_classes=2,
    base_channels=32,
    deep_supervision=False,
):
    model_name = model_name.lower()
    if model_name == "unet":
        return UNet(in_channels=in_channels, num_classes=num_classes, base_channels=base_channels)
    if model_name == "unetpp":
        return LiverECAUNetPlusPlus(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            deep_supervision=deep_supervision,
            use_eca=False,
            use_dilation=False,
        )
    if model_name == "eca-unetpp":
        return LiverECAUNetPlusPlus(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            deep_supervision=deep_supervision,
            use_eca=True,
            use_dilation=False,
        )
    if model_name == "eca-dilated-unetpp":
        return LiverECAUNetPlusPlus(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            deep_supervision=deep_supervision,
            use_eca=True,
            use_dilation=True,
        )
    raise ValueError(f"Unsupported model_name: {model_name}")
