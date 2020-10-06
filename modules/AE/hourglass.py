import torch
from torch import nn
from sys import exit as e


class DownBlock(nn.Module):
  def __init__(self, in_channel, out_channel):
    super(DownBlock, self).__init__()
    self.conv = nn.Conv2d(in_channels = in_channel, out_channels = out_channel, kernel_size = (3, 3), padding = 1)
    self.batch_norm = nn.BatchNorm2d(out_channel)
    self.relu = nn.ReLU()

  def forward(self, x):
    out = self.conv(x)
    out = self.batch_norm(out)
    out = self.relu(out)
    return out

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
    # out = self.avg_pool(out)
    return out



class Encoder(nn.Module):
  def __init__(self, in_channel, out_channel, feats):
    super(Encoder, self).__init__()
    encode_blocks = []
    for i in range(len(feats)-1):
      encode_blocks.append(UpBlock(feats[i], feats[i+1]))
    self.encode_blocks = nn.ModuleList(encode_blocks)

  def forward(self, x):
    outs = [x]
    for block in self.encode_blocks:
      outs.append(block(outs[-1]))
    return outs


class Decoder(nn.Module):
  def __init__(self, in_channel, out_channel, feats):
    super(Decoder, self).__init__()
    decode_blocks = []
    for i in range(len(feats)-1):
      decode_blocks.append(DownBlock(feats[i], feats[i+1]))
    self.decode_blocks = nn.ModuleList(decode_blocks)

  def forward(self, x):
    out = x.pop()
    for block in self.decode_blocks:
      out = block(out)
    return out



class Hourglass(nn.Module):
  def __init__(self, in_channel, out_channel, encoder_feats, decoder_feats):
    super(Hourglass, self).__init__()
    self.encoder = Encoder(in_channel, out_channel, encoder_feats)
    self.decoder = Decoder(in_channel, out_channel, decoder_feats)

  def forward(self, x):
    out = self.encoder(x)
    out = self.decoder(out)
    return out