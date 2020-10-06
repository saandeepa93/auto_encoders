import torch
from torch import nn
import numpy as np

from sys import exit as e


class DownBlock(nn.Module):
  def __init__(self, in_channel, out_channel, k, str, p):
    super(DownBlock, self).__init__()
    self.encoder = nn.Sequential(
      nn.Conv2d(in_channel, out_channel, kernel_size = k, stride = str, padding = p),
      nn.ReLU()
    )


  def forward(self, x):
    return self.encoder(x)


class UpBlock(nn.Module):
  def __init__(self, in_channel, out_channel, k, str, p):
    super(UpBlock, self).__init__()
    self.decoder = nn.Sequential(
      nn.ConvTranspose2d(in_channel, out_channel, kernel_size = k, stride = str, padding = p),
      nn.ReLU()
    )

  def forward(self, x):
    return self.decoder(x)


class Encoder(nn.Module):
  def __init__(self, in_channel, size, base_channel):
    super(Encoder, self).__init__()
    encoder_blocks = []
    for i in range(int(np.log2(size)) - 2):
      inc = in_channel if i==0 else (2**(i-1)) * base_channel
      outc = (2**(i)) * base_channel
      encoder_blocks.append(DownBlock(inc, outc, 4, 2, 1))
    else:
      self.final_channel = outc

    self.encoder_blocks = nn.ModuleList(encoder_blocks)
    self.final_encoder = nn.Conv2d(self.final_channel, self.final_channel, 4, 2, 0)


  def forward(self, x):
    out = x
    for block in self.encoder_blocks:
      out = block(out)
    out = self.final_encoder(out)
    h = out.view(out.size(0), -1)
    return h


class Decoder(nn.Module):
  def __init__(self, in_channel, out_channel, size):
    super(Decoder, self).__init__()
    decode_blocks = []
    inc = in_channel
    for i in range(int(np.log2(size)) - 2):
      inc = in_channel if i<=1 else in_channel//(2**(i-1))
      outc = in_channel if i==0 else (inc//2)
      k = 6 if i ==0 else 4
      st = 3 if i==0 else 2
      decode_blocks.append(UpBlock(inc, outc, k, st, 1))
    else:
      final_channel = outc
    self.decoder = nn.ModuleList(decode_blocks)
    self.final_layer = nn.ConvTranspose2d(final_channel, out_channel, kernel_size = 4, stride = 2, padding = 1)

  def forward(self, x):
    out = x
    for block in self.decoder:
      out = block(out)
    out = self.final_layer(out)
    out = torch.sigmoid(out)
    return out


class VAE_CNN(nn.Module):
  def __init__(self, in_channel, size, base_channel):
    super(VAE_CNN, self).__init__()

    z_dim = base_channel

    self.encoder = Encoder(in_channel, size, base_channel)
    h_dim = self.encoder.final_channel
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
    z, mu, logvar = self.reparameterize(h)
    z = self.latent(z)
    z = z.view(z.size(0), z.size(1), 1, 1)
    x = self.decoder(z)
    return x, mu, logvar