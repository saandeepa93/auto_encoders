import torch
from torch import nn
from sys import exit as e


class UpBlock(nn.Module):
  def __init__(self, in_channel, out_channel):
    super(UpBlock, self).__init__()
    self.conv = nn.ConvTranspose2d(in_channels = in_channel, out_channels = out_channel, kernel_size = (3, 3), padding = 1)
    self.batch_norm = nn.BatchNorm2d(out_channel)
    self.relu = nn.ReLU()
    self.avg_pool = nn.AvgPool2d((2, 2))

  def forward(self, x):
    out = self.conv(x)
    out = self.batch_norm(out)
    out = self.relu(out)
    out = self.avg_pool(out)
    return out



class Encoder(nn.Module):
  def __init__(self, in_channel, out_channel):
    super(Encoder, self).__init__()
    encode_blocks = []
    self.block1 = UpBlock(in_channel, 32)
    self.block2 = UpBlock(32, 64)
    self.block3 = UpBlock(64, 128)
    self.block4 = UpBlock(128, 64)
    self.block5 = UpBlock(64, out_channel)
    encode_blocks.append(self.block1)
    encode_blocks.append(self.block2)
    encode_blocks.append(self.block3)
    encode_blocks.append(self.block4)
    encode_blocks.append(self.block5)
    self.encode_blocks = nn.ModuleList(encode_blocks)

  def forward(self, x):
    outs = [x]
    for block in self.encode_blocks:
      outs.append(block(outs[-1]))
    return outs


class Hourglass(nn.Module):
  def __init__(self, in_channel, out_channel):
    super(Hourglass, self).__init__()
    self.encoder = Encoder(in_channel, out_channel)

  def forward(self, x):
    return self.encoder(x)