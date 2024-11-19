import torch
import torch.nn as nn
from config import swin_tiny_patch4_224_2 as swin
import torch.nn.functional as F
import math
# from .DFConv import DeformConv2d
import DFConv
import timm


class cSE(nn.Module):  # noqa: N801 c：

    def __init__(self, in_channels: int, r: int = 16):
        """
        Args:
            in_channels: The number of channels
                in the feature map of the input.
            r: The reduction ratio of the intermediate channels.
                Default: 16.
        """
        super().__init__()
        self.linear1 = nn.Linear(in_channels, in_channels // r)
        self.linear2 = nn.Linear(in_channels // r, in_channels)

    def forward(self, x: torch.Tensor):
        """Forward call."""
        input_x = x

        x = x.view(*(x.shape[:-2]), -1).mean(-1)
        x = F.relu(self.linear1(x), inplace=True)
        x = self.linear2(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = torch.sigmoid(x)

        x = torch.mul(input_x, x)
        return x


class sSE(nn.Module):
    def __init__(self, in_channels: int):
        """
        Args:
            in_channels: The number of channels
                in the feature map of the input.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor):
        """Forward call."""
        input_x = x

        x = self.conv(x)
        x = torch.sigmoid(x)

        x = torch.mul(input_x, x)
        return x


class scSE(nn.Module):  # noqa: N801

    def __init__(self, in_channels: int, r: int = 16):
        super().__init__()
        self.cse_block = cSE(in_channels, r)
        self.sse_block = sSE(in_channels)

    def forward(self, x: torch.Tensor):
        """Forward call."""
        cse = self.cse_block(x)
        sse = self.sse_block(x)
        x = torch.add(cse, sse)
        return x


class cSE1(nn.Module):  # noqa: N801

    def __init__(self, in_channels: int, r: int = 16):
        super().__init__()
        self.linear1 = nn.Linear(in_channels, in_channels // r)
        self.linear2 = nn.Linear(in_channels // r, in_channels)
        self.maxpool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x: torch.Tensor):
        """Forward call."""
        input_x = x
        x_max = self.maxpool(x)

        x_max = x_max.view(*(x_max.shape[:-2]))
        x_max = F.relu(self.linear1(x_max), inplace=True)
        x_max = self.linear2(x_max)

        x = x.view(*(x.shape[:-2]), -1).mean(-1)
        x = F.relu(self.linear1(x), inplace=True)
        x = self.linear2(x)

        x = torch.add(x, x_max)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = torch.sigmoid(x)

        x = torch.mul(input_x, x)
        return x


class sSE1(nn.Module):  # noqa: N801

    def __init__(self, in_channels: int):
        """
        Args:
            in_channels: The number of channels
                in the feature map of the input.
        """
        super().__init__()
        self.DWconv1 = DWconv(in_channels, 1, stride=1, padding=4, dilation=4)
        self.DWconv2 = DWconv(in_channels, 1, stride=1, padding=6, dilation=6)
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(3, 1, 1, 1)

    def forward(self, x: torch.Tensor):
        """Forward call."""
        input_x = x

        x1 = self.conv1(x)
        x2 = self.DWconv1(x)
        x3 = self.DWconv2(x)
        x = self.conv2(torch.cat([x1, x2, x3], dim=1))
        x = torch.sigmoid(x)
        x = torch.mul(input_x, x)
        return x


class scSE1(nn.Module):  # noqa: N801

    def __init__(self, in_channels: int, r: int = 16):
        super().__init__()
        self.cse_block = cSE(in_channels, r)
        self.sse_block = sSE1(in_channels)

    def forward(self, x: torch.Tensor):
        """Forward call."""
        cse = self.cse_block(x)
        sse = self.sse_block(x)
        x = torch.add(cse, sse)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class DWconv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, padding=1, dilation=1):
        super(DWconv, self).__init__()
        # 也相当于分组为1的分组卷积
        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=3,
                                    stride=stride,
                                    padding=padding,
                                    dilation=dilation,
                                    groups=in_ch)
        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out



class BiFFM(nn.Module):
    def __init__(self, ch_1, ch_2, ch_out, drop_rate=0.):
        super(BiFFM, self).__init__()


        self.drop_rate = drop_rate


        self.dropout = nn.Dropout2d(drop_rate)

        self.residual = nn.Sequential(
            nn.BatchNorm2d(ch_1 + ch_2),
            nn.ReLU(),
            nn.Conv2d(ch_1 + ch_2, ch_out, 3, 1, 1)
        )

    def forward(self, g, x):

        fuse = self.residual(torch.cat([g, x], 1))

        if self.drop_rate > 0:
            return self.dropout(fuse)
        else:
            return fuse



class ACO(nn.Module):
    def __init__(self, ch_1, ch_2, ch_out, if_PFN=False):
        super().__init__()
        self.ch_1 = ch_1
        self.pfn = if_PFN
        self.channel_att = cSE1(ch_1 + ch_2)
        self.conv1 = nn.Sequential(DWconv(ch_1 + ch_2, ch_1 + ch_2),
                                   nn.BatchNorm2d(ch_1 + ch_2),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(DWconv(ch_1 + ch_2, ch_1 + ch_2),
                                   nn.BatchNorm2d(ch_1 + ch_2),
                                   nn.ReLU())
        self.conv3 = nn.Conv2d(ch_1 + ch_2, ch_out, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        if self.pfn:
            self.final_mask = nn.Sequential(
                Conv(ch_1 + ch_2, ch_1 // 4, 3, bn=True, relu=True),
                Conv(ch_1 // 4, 1, 3, bn=False, relu=False)
            )
            self.final_boundary = nn.Sequential(
                Conv(ch_1 + ch_2, ch_1 // 4, 3, bn=True, relu=True),
                Conv(ch_1 // 4, 1, 3, bn=False, relu=False)
            )

        self.act = nn.ReLU()

    def forward(self, b, f):

        b_up = self.upsample(b)
        x_cat = torch.cat([b_up, f], dim=1)
        x = self.channel_att(x_cat)

        x_b = self.conv1(x)
        x_b = torch.add(x_cat, x_b)

        x_f = self.conv2(x)
        x_f = torch.add(x_cat, x_f)

        if self.pfn:
            mask = x_b
            boundary = x_b
            mask = self.final_mask(mask)
            boundary = self.final_boundary(boundary)
            x_f = self.conv3(x_f)
            return x_f, mask, boundary
        x_f = self.conv3(x_f)
        return x_f


class CTCFNet(nn.Module):
    def __init__(self, num_classes=6, drop_rate=0.4, normal_init=True, pretrained=False):
        super(CTCFNet, self).__init__()

        # self.efficienet = timm.create_model('efficientnet_b3')
        # self.act1 = nn.SiLU()
        # if pretrained:
        #     self.efficienet.load_state_dict(torch.load('./pretrained/efficientnet_b3_ra2-cf984f9c.pth'))

        self.efficienet = timm.create_model('efficientnetv2_rw_t', num_classes=0)
        self.act1 = nn.SiLU()
        if pretrained:
            self.efficienet = timm.create_model('efficientnetv2_rw_t.ra2_in1k', num_classes=0)

        self.transformer = swin(pretrained=pretrained)
        self.extract_features = {}

        self.final_x = nn.Sequential(
            Conv(256, 64, 1, bn=True, relu=True),
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
        )

        self.final_1 = nn.Sequential(
            Conv(128, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
        )

        self.final_2 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
        )

        self.final_3 = nn.Sequential(
            Conv(32, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
        )

        self.up_c = BiFFM(ch_1=208, ch_2=768, ch_out=256, drop_rate=drop_rate / 2)
        self.up_c_1_1 = BiFFM(ch_1=128, ch_2=384, ch_out=128, drop_rate=drop_rate / 2)
        self.up_c_1_2 = ACO(ch_1=256, ch_2=128, ch_out=128, if_PFN=True)
        self.up_c_2_1 = BiFFM(ch_1=48, ch_2=192, ch_out=64, drop_rate=drop_rate / 2)
        self.up_c_2_2 = ACO(ch_1=128, ch_2=64, ch_out=64, if_PFN=False)
        ###
        self.up_c_3_1 = BiFFM(ch_1=40, ch_2=96, ch_out=32, drop_rate=drop_rate / 2)
        self.up_c_3_2 = ACO(ch_1=64, ch_2=32, ch_out=32, if_PFN=True)

        self.drop = nn.Dropout2d(drop_rate)

        if normal_init:
            self.init_weights()

    def forward(self, imgs):
        # transformer path
        x_b = self.transformer(imgs)
        x_b_1 = x_b[0]
        x_b_1 = torch.transpose(x_b_1, 1, 2)
        x_b_1 = x_b_1.view(x_b_1.shape[0], -1, 64, 64)
        x_b_1 = self.drop(x_b_1)

        x_b_2 = x_b[1]
        x_b_2 = torch.transpose(x_b_2, 1, 2)
        x_b_2 = x_b_2.view(x_b_2.shape[0], -1, 32, 32)
        x_b_2 = self.drop(x_b_2)

        x_b_3 = x_b[2]
        x_b_3 = torch.transpose(x_b_3, 1, 2)
        x_b_3 = x_b_3.view(x_b_3.shape[0], -1, 16, 16)
        x_b_3 = self.drop(x_b_3)

        x_b_4 = x_b[3]
        x_b_4 = torch.transpose(x_b_4, 1, 2)
        x_b_4 = x_b_4.view(x_b_4.shape[0], -1, 8, 8)
        x_b_4 = self.drop(x_b_4)

        # CNN path
        ####effinetb3
        x_u128 = self.efficienet.conv_stem(imgs)
        x_u128 = self.efficienet.bn1(x_u128)
        x_u128 = self.act1(x_u128)
        x_u128 = self.efficienet.blocks[0](x_u128)
        x_u64 = self.efficienet.blocks[1](x_u128)

        x_u_2 = self.efficienet.blocks[2](x_u64)
        x_u_2 = self.drop(x_u_2)

        x_u_3 = self.efficienet.blocks[3](x_u_2)
        x_u_3 = self.drop(x_u_3)

        x_u_3 = self.efficienet.blocks[4](x_u_3)
        x_u_3 = self.drop(x_u_3)

        x_u = self.efficienet.blocks[5](x_u_3)
        x_u = self.drop(x_u)

        # joint path
        x_c = self.up_c(x_u, x_b_4)
        x_c_1_1 = self.up_c_1_1(x_u_3, x_b_3)
        x_c_1, mask_1, bound1 = self.up_c_1_2(x_c, x_c_1_1)

        x_c_2_1 = self.up_c_2_1(x_u_2, x_b_2)
        x_c_2 = self.up_c_2_2(x_c_1, x_c_2_1)

        ###
        x_c_3_1 = self.up_c_3_1(x_u64, x_b_1)
        x_c_3, mask_2, bound2 = self.up_c_3_2(x_c_2, x_c_3_1)


        map_x_ = self.final_x(x_c)
        map_1_ = self.final_1(x_c_1)
        map_2_ = self.final_2(x_c_2)
        map_3_ = self.final_3(x_c_3)

        self.extract_features['fuse1'] = F.softmax(map_x_)
        self.extract_features['fuse2'] = F.softmax(map_1_)
        self.extract_features['fuse3'] = F.softmax(map_2_)
        self.extract_features['fuse4'] = F.softmax(map_3_)

        #
        map_x = F.interpolate(map_x_, scale_factor=32, mode='bilinear')
        map_1 = F.interpolate(map_1_, scale_factor=16, mode='bilinear')
        map_2 = F.interpolate(map_2_, scale_factor=8, mode='bilinear')
        map_3 = F.interpolate(map_3_, scale_factor=4, mode='bilinear')

        map = map_x + map_1 + map_2 + map_3
        self.extract_features['fuse'] = F.softmax(map)

        mask_1 = F.interpolate(mask_1, scale_factor=16, mode='bilinear')
        mask_2 = F.interpolate(mask_2, scale_factor=4, mode='bilinear')

        bound1 = F.interpolate(bound1, scale_factor=16, mode='bilinear')
        bound2 = F.interpolate(bound2, scale_factor=4, mode='bilinear')

        return map, mask_1, mask_2, bound1, bound2

    def init_weights(self):
        self.final_x.apply(init_weights)
        self.final_1.apply(init_weights)
        self.final_2.apply(init_weights)
        self.final_3.apply(init_weights)
        self.up_c.apply(init_weights)
        self.up_c_1_1.apply(init_weights)
        self.up_c_1_2.apply(init_weights)
        self.up_c_2_1.apply(init_weights)
        self.up_c_2_2.apply(init_weights)
        self.up_c_3_1.apply(init_weights)
        self.up_c_3_2.apply(init_weights)


def init_weights(m):
    """
    Initialize weights of layers using Kaiming Normal (He et al.) as argument of "Apply" function of
    "nn.Module"
    :param m: Layer to initialize
    :return: None
    """
    if isinstance(m, nn.Conv2d):
        '''
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        trunc_normal_(m.weight, std=math.sqrt(1.0/fan_in)/.87962566103423978)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
        '''
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # num_channels = groups * channels_per_group

    # grouping, 通道分组
    # b, num_channels, h, w =======>  b, groups, channels_per_group, h, w
    x = x.view(batchsize, groups, channels_per_group, height, width)

    # channel shuffle, 通道洗牌
    x = torch.transpose(x, 1, 2).contiguous()
    # x.shape=(batchsize, channels_per_group, groups, height, width)
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.identity = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.double_conv(x) + self.identity(x))


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)

        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x