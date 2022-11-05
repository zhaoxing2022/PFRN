import torch
import torch.nn as nn
from torchvision.models import resnet50

import torch.nn.functional as F


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        resnet = resnet50()
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        out1 = self.relu(self.bn1(self.conv1(x)))
        out1_ = self.maxpool(out1)
        out2 = self.layer1(out1_)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out1, out2, out3, out4, out5


class UpsampleModule(nn.Module):
    def __init__(self, in_ch):
        super(UpsampleModule, self).__init__()
        self.dec_conv = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU(True)

    def forward(self, input, assign=None):
        output = F.interpolate(input, [assign, assign], mode='bilinear')
        output = self.dec_conv(output)
        return self.relu(self.bn(output))


class Con3x3WithBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Con3x3WithBnRelu, self).__init__()
        self.con3x3 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(True)

    def forward(self, input):
        return self.relu(self.bn(self.con3x3(input)))


class GPEA(nn.Module):
    def __init__(self, in_ch):
        super(GPEA, self).__init__()
        self.up = UpsampleModule(in_ch)
        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

    def forward(self, high, low, assign=None):
        high = self.up(high, assign)
        atten = self.conv1(high) * self.conv2(low)
        return atten * low + low


class ORA(nn.Module):
    def __init__(self, in_ch):
        super(ORA, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

    def forward(self, conv_ori, deconv_ori):
        conv = self.conv1(conv_ori)
        deconv = self.conv2(deconv_ori)
        value = self.conv3(conv_ori)
        atten = conv * deconv

        return atten * value


class Con1x1WithBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Con1x1WithBnRelu, self).__init__()
        self.con1x1 = nn.Conv2d(in_ch, out_ch,
                                kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(True)

    def forward(self, input):
        return self.relu(self.bn(self.con1x1(input)))


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.resnet = ResNet()

        self.deconv_redcue = Con1x1WithBnRelu(2048, 64)

        # 8->16
        self.deconv1 = UpsampleModule(64)
        self.deconv1_reduce = Con1x1WithBnRelu(1024, 64)
        self.deconv1_conv3x3 = Con3x3WithBnRelu(64 + 64, 64)

        # 16->32
        self.deconv2 = UpsampleModule(64)
        self.deconv2_reduce = Con1x1WithBnRelu(512, 64)
        self.deconv2_conv3x3 = Con3x3WithBnRelu(64 + 64, 64)

        # 32->64
        self.deconv3 = UpsampleModule(64)
        self.deconv3_reduce = Con1x1WithBnRelu(256, 64)
        self.deconv3_conv3x3 = Con3x3WithBnRelu(64 + 64, 64)

        # 64->128
        self.deconv4 = UpsampleModule(64)
        self.deconv4_reduce = Con1x1WithBnRelu(64, 64)
        self.deconv4_conv3x3 = Con3x3WithBnRelu(64 + 64, 64)

        self.GPEA_16x16 = GPEA(64)
        self.GPEA2_32x32 = GPEA(64)
        self.GPEA3_64x64 = GPEA(64)
        self.GPEA4_128x128 = GPEA(64)

        self.ORA_128x128 = ORA(64)
        self.ORA_64x64 = ORA(64)
        self.ORA_32x32 = ORA(64)
        self.ORA_16x16 = ORA(64)

        self.s2_deconv128 = Con3x3WithBnRelu(64 * 2, 64)
        self.s2_deconv64 = Con3x3WithBnRelu(64 * 2, 64)
        self.s2_deconv32 = Con3x3WithBnRelu(64 * 2, 64)
        self.s2_deconv16 = Con3x3WithBnRelu(64 * 2, 64)

        self.s2_final_conv = nn.Sequential(
            nn.Conv2d(64 * 4, 64 * 2, 3, padding=1, bias=False), nn.BatchNorm2d(64 * 2), nn.ReLU(),
            nn.Conv2d(64 * 2, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        )


    def forward(self, x):
        #The feature size is based on inputs (B, C, H, W)--> (B, 3, 256, 256)
        conv1_128x128, conv2_64x64, conv3_32x32, conv4_16x16, conv5_8x8 = self.resnet(x)

        conv5_8x8_reduce = self.deconv_redcue(conv5_8x8)
        conv4_16x16_reduce = self.deconv1_reduce(conv4_16x16)
        conv32x32_reduce = self.deconv2_reduce(conv3_32x32)
        conv64x64_reduce = self.deconv3_reduce(conv2_64x64)
        conv128x128_reduce = self.deconv4_reduce(conv1_128x128)

        deconv16x16 = self.deconv1(conv5_8x8_reduce, conv4_16x16_reduce.size(2))
        GPEA_16x16 = self.GPEA_16x16(conv5_8x8_reduce, conv4_16x16_reduce, conv4_16x16_reduce.size(2))
        deconv16x16 = self.deconv1_conv3x3(torch.cat((deconv16x16, GPEA_16x16), dim=1))

        deconv32x32 = self.deconv2(deconv16x16, conv32x32_reduce.size(2))
        GPEA2_32x32 = self.GPEA2_32x32(conv5_8x8_reduce, conv32x32_reduce, conv32x32_reduce.size(2))
        deconv32x32 = self.deconv2_conv3x3(torch.cat((deconv32x32, GPEA2_32x32), dim=1))

        deconv64x64 = self.deconv3(deconv32x32, conv64x64_reduce.size(2))
        GPEA3_64x64 = self.GPEA3_64x64(conv5_8x8_reduce, conv64x64_reduce, conv64x64_reduce.size(2))
        deconv64x64 = self.deconv3_conv3x3(torch.cat((deconv64x64, GPEA3_64x64), dim=1))

        deconv128x128 = self.deconv4(deconv64x64, conv128x128_reduce.size(2))
        GPEA4_128x128 = self.GPEA4_128x128(conv5_8x8_reduce, conv128x128_reduce, conv128x128_reduce.size(2))
        deconv128x128 = self.deconv4_conv3x3(torch.cat((deconv128x128, GPEA4_128x128), dim=1))

        ORA_128x128 = self.ORA_128x128(conv128x128_reduce, deconv128x128)
        s2_deconv128x128 = self.s2_deconv128(torch.cat((ORA_128x128, deconv128x128), dim=1))

        ORA_64x64 = self.ORA_64x64(conv64x64_reduce, deconv64x64)
        s2_deconv64x64 = self.s2_deconv64(torch.cat((ORA_64x64, deconv64x64), dim=1))

        ORA_32x32 = self.ORA_32x32(conv32x32_reduce, deconv32x32)
        s2_deconv32x32 = self.s2_deconv32(torch.cat((ORA_32x32, deconv32x32), dim=1))

        ORA_16x16 = self.ORA_16x16(conv4_16x16_reduce, deconv16x16)
        s2_deconv16x16 = self.s2_deconv16(torch.cat((ORA_16x16, deconv16x16), dim=1))

        up1 = torch.nn.functional.interpolate(s2_deconv16x16, x.size()[2:], mode="bilinear")
        up2 = torch.nn.functional.interpolate(s2_deconv32x32, x.size()[2:], mode="bilinear")
        up3 = torch.nn.functional.interpolate(s2_deconv64x64, x.size()[2:], mode="bilinear")
        up4 = torch.nn.functional.interpolate(s2_deconv128x128, x.size()[2:], mode="bilinear")

        result = self.s2_final_conv(torch.cat((up1, up2, up3, up4), dim=1))

        return torch.sigmoid(result), None, None, None, None