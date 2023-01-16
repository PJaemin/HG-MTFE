import torch
import torch.nn as nn
import torch.nn.functional as F
from unet_model import UNet
from Transformer import TransformerEncoder



class intensityTransform(nn.Module):
    def __init__(self, intensities, channels, **kwargs):
        super(intensityTransform, self).__init__(**kwargs)
        self.channels = channels
        self.scale = intensities - 1

    def get_config(self):
        config = super(intensityTransform, self).get_config()
        config.update({'channels': self.channels, 'scale': self.scale})
        return config

    def forward(self, inputs):
        images, transforms = inputs

        transforms = transforms.unsqueeze(3)  # Index tensor must have the same number of dimensions as input tensor

        # images = 0.5 * images + 0.5
        images = torch.round(self.scale * images)
        images = images.type(torch.LongTensor)
        images = images.cuda()
        transforms = transforms.cuda()
        minimum_w = images.size(3)
        iter_n = 0
        temp = 1
        while minimum_w > temp:
            temp *= 2
            iter_n += 1

        for i in range(iter_n):
            transforms = torch.cat([transforms, transforms], dim=3)

        images = torch.split(images, 1, dim=1)
        transforms = torch.split(transforms, 1, dim=1)

        x = torch.gather(input=transforms[0], dim=2, index=images[0])
        y = torch.gather(input=transforms[1], dim=2, index=images[1])
        z = torch.gather(input=transforms[2], dim=2, index=images[2])

        xx = torch.cat([x, y, z], dim=1)

        return xx


class conv_block(nn.Module):
    def __init__(self, input_ch, output_ch, kernel_size, strides, dropout_rate=0.1):
        super(conv_block, self).__init__()
        self.dropout_rate = dropout_rate
        padding = kernel_size//2
        self.cb_conv1 = nn.Conv2d(input_ch, output_ch, kernel_size, strides, padding=padding, bias=False)
        self.cb_batchNorm = nn.BatchNorm2d(output_ch)
        if dropout_rate > 0:
            self.dropout = nn.Dropout2d(p=dropout_rate)

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x):
        x = self.cb_conv1(x)
        x = self.cb_batchNorm(x)
        x = self.swish(x)
        if self.dropout_rate > 0:
            x = self.dropout(x)
        return x


class SFC_module(nn.Module):
    def __init__(self, in_ch, out_ch, expansion, num):
        super(SFC_module, self).__init__()
        exp_ch = int(in_ch * expansion)
        if num == 1:
            self.se_conv = nn.Conv2d(in_ch, exp_ch, 3, 1, 1, groups=in_ch)
        else:
            self.se_conv = nn.Conv2d(in_ch, exp_ch, 3, 2, 1, groups=in_ch)
        self.se_bn = nn.BatchNorm2d(exp_ch)
        self.se_relu = nn.ReLU()
        self.hd_conv = nn.Conv2d(exp_ch, exp_ch, 3, 1, 1, groups=in_ch)
        self.hd_bn = nn.BatchNorm2d(exp_ch)
        self.hd_relu = nn.ReLU()
        self.cp_conv = nn.Conv2d(exp_ch, out_ch, 1, 1, groups=in_ch)
        self.cp_bn = nn.BatchNorm2d(out_ch)
        self.pw_conv = nn.Conv2d(out_ch, out_ch, 1, 1)
        self.pw_bn = nn.BatchNorm2d(out_ch)
        self.pw_relu = nn.ReLU()
        self.ca_gap = nn.AdaptiveAvgPool2d(1)
        self.ca_map = nn.AdaptiveMaxPool2d(1)
        self.ca_conv = nn.Conv1d(1, 1, 3, 1, 1)
        self.ca_sig = nn.Sigmoid()

    def forward(self, x):
        x = self.se_conv(x)
        x = self.se_bn(x)
        x = self.se_relu(x)
        x = self.hd_conv(x)
        x = self.hd_bn(x)
        x = self.hd_relu(x)
        x = self.cp_conv(x)
        x = self.cp_bn(x)
        x = self.pw_conv(x)
        x = self.pw_bn(x)
        x = self.pw_relu(x)

        aa = self.ca_gap(x)
        ma = self.ca_map(x)
        aa = aa.squeeze(3)
        aa = aa.permute(0, 2, 1)
        ma = ma.squeeze(3)
        ma = ma.permute(0, 2, 1)
        aa = self.ca_conv(aa)
        ma = self.ca_conv(ma)
        a = aa + ma
        a = a.permute(0, 2, 1)
        a = self.ca_sig(a)
        a = a.unsqueeze(3)
        x = x * a
        return x


class HSFC_module(nn.Module):
    def __init__(self, in_ch, expansion):
        super(HSFC_module, self).__init__()
        exp_ch = int(in_ch * expansion)
        self.se_conv = nn.Conv1d(in_ch, exp_ch, 3, 1, 1, groups=in_ch)
        self.se_bn = nn.BatchNorm1d(exp_ch)
        self.se_relu = nn.ReLU()
        self.hd_conv = nn.Conv1d(exp_ch, exp_ch, 3, 1, 1, groups=in_ch)
        self.hd_bn = nn.BatchNorm1d(exp_ch)
        self.hd_relu = nn.ReLU()
        self.cp_conv = nn.Conv1d(exp_ch, in_ch, 1, 1, groups=in_ch)
        self.cp_bn = nn.BatchNorm1d(in_ch)
        self.pw_conv = nn.Conv1d(in_ch, in_ch, 1, 1)
        self.pw_bn = nn.BatchNorm1d(in_ch)
        self.pw_relu = nn.ReLU()

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x):
        x = self.se_conv(x)
        x = self.se_bn(x)
        x = self.se_relu(x)
        x = self.hd_conv(x)
        x = self.hd_bn(x)
        x = self.hd_relu(x)
        x = self.cp_conv(x)
        x = self.cp_bn(x)
        x = self.pw_conv(x)
        x = self.pw_bn(x)
        x = self.pw_relu(x)
        return x


class Histogram_network(nn.Module):

    def __init__(self):
        super(Histogram_network, self).__init__()
        expansion = 4
        C = 24

        self.stage1 = HSFC_module(3, expansion)
        self.stage2 = HSFC_module(3, expansion)
        self.stage3 = HSFC_module(3, expansion)
        self.stage4 = HSFC_module(3, expansion)


    def forward(self, h):
        y = self.stage1(h)
        y = self.stage2(y)
        y = self.stage3(y)
        y = self.stage4(y)
        y = y.flatten(1)

        return y

class Attention_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Attention_block, self).__init__()
        self.g_conv = nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=int(in_ch/2))
        self.g_bn = nn.BatchNorm2d(in_ch)
        self.g_relu = nn.ReLU()

        self.pw_conv = nn.Conv2d(in_ch, out_ch, 1, 1)
        self.pw_bn = nn.BatchNorm2d(out_ch)
        self.pw_relu = nn.ReLU()

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, in1, in2):
        i1_1, i1_2, i1_3 = torch.chunk(in1, 3, dim=1)
        i2_1, i2_2, i2_3 = torch.chunk(in2, 3, dim=1)
        x = torch.cat([i1_1, i2_1, i1_2, i2_2, i1_3, i2_3], dim=1)

        x = self.g_conv(x)
        x = self.g_bn(x)
        x = self.g_relu(x)

        x = self.pw_conv(x)
        x = self.pw_bn(x)
        x = self.pw_relu(x)

        return x


class Simpconv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride=1, padding=0):
        super(Simpconv2d, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        y = self.relu(y)
        return y


class Bottleneck(nn.Module):
    def __init__(self):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(2, 1, 1, 1)
        self.bn1 = nn.BatchNorm1d(1)
        self.relu1 = nn.ReLU()
        self.conv2_1 = nn.Conv1d(1, 1, 3, 1, 1)
        self.bn2_1 = nn.BatchNorm1d(1)
        self.relu2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv1d(1, 1, 3, 1, 1)
        self.bn2_2 = nn.BatchNorm1d(1)
        self.relu2_2 = nn.ReLU()

    def forward(self, x1, x2):
        x = self.conv1(torch.cat((x1, x2), dim=1))
        x = self.bn1(x)
        x = self.relu1(x)
        w1 = self.conv2_1(x)
        w1 = self.bn2_1(w1)
        w1 = self.relu2_1(w1)
        y1 = x1 * w1
        w2 = self.conv2_2(x)
        w2 = self.bn2_2(w2)
        w2 = self.relu2_2(w2)
        y2 = x2 * w2
        return y1, y2


class Image_network(nn.Module):

    def __init__(self):
        super(Image_network, self).__init__()
        expansion = 4
        C = 6

        self.WM_gen = UNet(12, 3)
        # self.inputfus = Attention_block(6,6)

        self.stage1 = Simpconv2d(3, C, 3, 1, 1)
        self.stage2 = SFC_module(C, 2 * C, expansion, 1)
        self.stage3 = SFC_module(2 * C, 4 * C, expansion, 2)
        self.stage4 = SFC_module(4 * C, 8 * C, expansion, 3)
        self.stage5 = SFC_module(8 * C, 16 * C, expansion, 4)
        self.stage6 = SFC_module(16 * C, 32 * C, expansion, 5)
        self.stage7 = SFC_module(32 * C, 64 * C, expansion, 6)
        self.stage8 = SFC_module(64 * C, 128 * C, expansion, 7)
        self.stage9 = Simpconv2d(128 * C, 2304, 1, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.r_conv = Bottleneck()
        # self.g_conv = Bottleneck()
        # self.b_conv = Bottleneck()

        self.TE1 = TransformerEncoder(768, 1)
        # self.TE2 = TransformerEncoder(256, 1)
        # self.TE3 = TransformerEncoder(256, 1)
        # self.FC1 = nn.Linear(768, 768)
        # self.FC2 = nn.Linear(768, 768)
        # self.FC3 = nn.Linear(768, 768)

        # self.MHA = MultiHeadAtt(256, 256)
        # self.Norm = nn.LayerNorm(256)
        # self.MLP = MultiLinearPerceptron(768, 1)

        self.intensity_trans = intensityTransform(intensities=256, channels=3)

    def forward(self, x, hist):
        hist = torch.flatten(hist, start_dim=1)
        # hist = hist.unsqueeze(1)

        x_256 = F.interpolate(x, 256)
        y = self.stage1(x_256)
        # y = self.stage1_bn(y)
        # y = self.stage1_af(y)

        y = self.stage2(y)
        y = self.stage3(y)
        y = self.stage4(y)
        y = self.stage5(y)
        y = self.stage6(y)
        y = self.stage7(y)
        y = self.stage8(y)
        y = self.stage9(y)
        y = self.gap(y)
        y = y.squeeze(2)
        y = y.squeeze(2)
        y = y.unsqueeze(1)
        r, g, b = torch.chunk(y, 3, dim=2)
        k_r, h_r = self.r_conv(r, hist.unsqueeze(1))
        k_g, h_g = self.r_conv(g, hist.unsqueeze(1))
        k_b, h_b = self.r_conv(b, hist.unsqueeze(1))

        tf1, tf2, tf3 = self.TE1(r, g, b, h_r, h_g, h_b, k_r, k_g, k_b)
        # print(tf1.shape)
        # print(tf2.shape)

        tf1 = torch.sigmoid(tf1.unsqueeze(1))
        tf1 = torch.cat((torch.chunk(tf1, 3, dim=2)), dim=1)
        tf2 = torch.sigmoid(tf2.unsqueeze(1))
        tf2 = torch.cat((torch.chunk(tf2, 3, dim=2)), dim=1)
        tf3 = torch.sigmoid(tf3.unsqueeze(1))
        tf3 = torch.cat((torch.chunk(tf3, 3, dim=2)), dim=1)

        xy1 = self.intensity_trans((x, tf1))
        xy2 = self.intensity_trans((x, tf2))
        xy3 = self.intensity_trans((x, tf3))

        w = self.WM_gen(torch.cat((x, xy1, xy2, xy3), dim=1))
        w = torch.sigmoid(w)
        w1, w2, w3 = torch.chunk(w, 3, dim=1)
        # print(w1)

        wm1 = w1 / (w1 + w2 + w3)
        wm2 = w2 / (w1 + w2 + w3)
        wm3 = w3 / (w1 + w2 + w3)

        xy = wm1 * xy1 + wm2 * xy2 + wm3 * xy3

        return xy, (tf1, tf2, tf3), (wm1, wm2, wm3), (xy1, xy2, xy3)

