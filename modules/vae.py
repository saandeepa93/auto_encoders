import torch
from torch import nn
from torch.nn import functional as F


class VAE(nn.Module):
  def __init__(self):
    super(VAE, self).__init__()
    self.encode1 = nn.Linear(784, 400)
    self.mu = nn.Linear(400, 20)
    self.sigma = nn.Linear(400, 20)

    self.decode1 = nn.Linear(20, 400)
    self.decode2 = nn.Linear(400, 784)

  def encode(self, x):
    h1 = F.relu(self.encode1(x))
    return self.mu(h1), self.sigma(h1)

  def decode(self, z):
    d1 = F.relu(self.decode1(z))
    return torch.sigmoid(self.decode2(d1))

  def reparameterize(self, mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.rand_like(std)
    return mu + eps * std

  def forward(self, x):
    mu, logvar = self.encode(x.view(-1, 784))
    z = self.reparameterize(mu, logvar)
    return self.decode(z), mu, logvar
