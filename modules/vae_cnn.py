import torch
from torch import nn
import numpy as np

from sys import exit as e


class DownBlock(nn.Module):
  def __init__(self, in_channel, out_channel):
    super(DownBlock, self).__init__()
    self.encoder = nn.Sequential(
      nn.Conv2d(in_channel, out_channel, kernel_size = 4, stride = 2),
      nn.ReLU()
    )


  def forward(self, x):
    return self.encoder(x)


class UpBlock(nn.Module):
  def __init__(self, in_channel, out_channel):
    super(UpBlock, self).__init__()
    self.decoder = nn.Sequential(
      nn.ConvTranspose2d(in_channel, out_channel, kernel_size = 5, stride = 2),
      nn.ReLU()
    )

  def forward(self, x):
    return self.decoder(x)


class Encoder(nn.Module):
  def __init__(self, in_channel, size, base_channel):
    super(Encoder, self).__init__()
    encode_blocks = []
    for i in range(int(np.log2(size)) - 2):
      inc = in_channel if i==0 else (2**(i-1)) * base_channel
      outc = base_channel if i==0 else (2 ** (i)) * base_channel
      encode_blocks.append(DownBlock(inc, outc))
    else:
      self.final_encode = outc
    self.encoder = nn.ModuleList(encode_blocks)
    self.final_layer = nn.Conv2d(outc, outc, kernel_size = (2, 2), stride = 1)


  def forward(self, x):
    out = x
    print(out.size())
    for block in self.encoder:
      out = block(out)
      print(out.size())
    out = self.final_layer(out)
    print(out.size())
    print("-"*20)
    h = out.view(out.size(0), -1)
    return h


class Decoder(nn.Module):
  def __init__(self, in_channel, out_channel, size):
    super(Decoder, self).__init__()
    decode_blocks = []
    inc = in_channel
    for i in range(int(np.log2(size)) - 3):
      inc = in_channel if i==0 else in_channel//(2**i)
      outc = (inc//2)
      decode_blocks.append(UpBlock(inc, outc))
    else:
      final_channel = outc
    self.decoder = nn.ModuleList(decode_blocks)
    self.final_layer = nn.ConvTranspose2d(final_channel, out_channel, kernel_size = 6, stride = 2)

  def forward(self, x):
    out = x
    print(out.size())
    for block in self.decoder:
      out = block(out)
      print(out.size())
    out = self.final_layer(out)
    print(out.size())
    e()
    out = torch.sigmoid(out)
    return out


class VAE_CNN(nn.Module):
  def __init__(self, in_channel, size, base_channel):
    super(VAE_CNN, self).__init__()

    z_dim = base_channel

    self.encoder = Encoder(in_channel, size, base_channel)
    h_dim = self.encoder.final_encode
    self.mu = nn.Linear(h_dim, z_dim)
    self.sigma = nn.Linear(h_dim, z_dim)

    self.latent = nn.Linear(z_dim, h_dim)
    self.decoder = Decoder(h_dim, in_channel, size)

  def reparameterize(self, h):
    mu, logvar = self.mu(h), self.sigma(h)
    std = logvar.mul(0.5).exp_()
    esp = torch.randn(*mu.size())
    z = mu + std * esp
    return z, mu, logvar

  def forward(self, x):
    h = self.encoder(x)
    print(h.size())
    z, mu, logvar = self.reparameterize(h)
    print(z.size())
    z = self.latent(z)
    print(z.size())
    z = z.view(z.size(0), z.size(1), 1, 1)

    x = self.decoder(z)
    print(x.size())
    e()
    return x, mu, logvar