import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import numpy as np
import math
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm, MultiheadAttention

from networks.transnext import transnext_base
class ConvBlock(nn.Module):


    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x


class Bridge(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels),
        )

    def forward(self, x):
        return self.bridge(x)



class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None, upsampling_method="conv_transpose", islast=False):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels

        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if islast:
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=4, stride=4)
        else:
            if upsampling_method == "conv_transpose":
                self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
            elif upsampling_method == "bilinear":
                self.upsample = nn.Sequential(
                    nn.Upsample(mode='bilinear', scale_factor=2),
                    nn.Conv2d(up_conv_in_channels, up_conv_out_channels, kernel_size=1, stride=1)
                )
        self.conv_block_1 = ConvBlock(in_channels, out_channels, kernel_size=3,padding=1,stride=1)
        self.conv_block_2 = ConvBlock(out_channels, out_channels, kernel_size=3,padding=1,stride=1)

    def forward(self, up_x, down_x):

        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)

        return x


class CGblock(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.conv_block_1 = ConvBlock(self.in_channels + self.n_classes, self.in_channels)

        self.channelAtten = ChannelAttention(in_channels)
        self.text = nn.Linear(768, self.in_channels)

    def forward(self, x, x_text):

        if self.in_channels != 2048:
            x_text = self.text(x_text)

        imshape = x.shape
        image_features = x
        image_features = image_features.permute(0, 2, 3, 1).reshape(-1, self.in_channels)
        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = x_text / x_text.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        out = logits_per_image.float().view(imshape[0], imshape[2], imshape[3], -1).permute(0,3,1,2)
        out = torch.concat([x,out],dim=1)
        out = self.conv_block_1(out)
        out = self.channelAtten(out)

        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * x

class ImageBranch(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()

        self.encoder = transnext_base(n_classes)
        state_dict = torch.load('pretrained_ckpt/transnext_base_224_1k.pth', map_location='cuda')
        state_dict.pop('head.weight', None)
        state_dict.pop('head.bias', None)
        # 列出要忽略的层
        layers_to_ignore = [
            'block4.0.attn.qkv.weight', 'block4.0.attn.qkv.bias', 'block4.0.attn.proj.weight',
            'block4.1.attn.qkv.weight', 'block4.1.attn.qkv.bias', 'block4.1.attn.proj.weight',
            'block4.2.attn.qkv.weight', 'block4.2.attn.qkv.bias', 'block4.2.attn.proj.weight',
            'block4.3.attn.qkv.weight', 'block4.3.attn.qkv.bias', 'block4.3.attn.proj.weight',
            'block4.4.attn.qkv.weight', 'block4.4.attn.qkv.bias', 'block4.4.attn.proj.weight'
        ]

        # 删除 state_dict 中要忽略的层
        for layer in layers_to_ignore:
            if layer in state_dict:
                del state_dict[layer]
        missing_keys, unexpected_keys = self.encoder.load_state_dict(state_dict, strict=False)

        if not missing_keys and not unexpected_keys:
            print("All keys matched successfully and weights are loaded properly.")
        else:
            if missing_keys:
                print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys}")

        up_blocks = []
        self.n_classes = n_classes

        self.bridge = Bridge(768, 768)
        up_blocks.append(UpBlock(768, 384))
        up_blocks.append(UpBlock(384, 192))
        up_blocks.append(UpBlock(192, 96))
        up_blocks.append(UpBlock(in_channels=48 + 3, out_channels=48,
                                                    up_conv_in_channels=96, up_conv_out_channels=48, islast=True))

        self.up_blocks = nn.ModuleList(up_blocks)
        self.cgblock = CGblock(48, self.n_classes)
        self.out = nn.Conv2d(48, n_classes, kernel_size=1, stride=1)

    def forward(self, x, x_text):
        x_text = x_text.squeeze(-1).squeeze(-1)
        x = x.to('cuda')
        x, downsample = self.encoder(x)
        x = self.bridge(x)
        for i, block in enumerate(self.up_blocks):
            x = block(x, downsample[3-i])
        x = self.cgblock(x, x_text)

        x = self.out(x)

        return x

