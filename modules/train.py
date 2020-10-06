import torch
from torch import optim
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from sys import exit as e

from modules.vae import VAE
from modules.vae_cnn import VAE_CNN
from modules.util import vae_loss
from modules.dataset import Dset


def train_data(configs):
  lr = float(configs["params"]["lr"])
  epochs = float(configs["params"]["epochs"])
  img_size = int(configs["params"]["img_size"])
  in_channel = int(configs["params"]["in_channel"])
  base_channel = int(configs["params"]["base_channel"])
  root_dir = configs["paths"]["root_dir"]
  ext = configs["params"]["ext"]
  b_size = int(configs["params"]["b_size"])

  #BP4D
  dataset = Dset(root_dir, img_size, ext)
  train_loader = DataLoader(dataset, b_size, shuffle=False)

  #MNIST
  # transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
  # train_dataset = torchvision.datasets.MNIST(
  #   root="~/torch_datasets", train=True, transform=transform, download=True
  # )
  # train_loader = torch.utils.data.DataLoader(
  #     train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
  # )


  model = VAE_CNN(in_channel, img_size, base_channel)
  optimizer = optim.Adam(model.parameters(), lr=lr)

  for i in range(int(epochs)):
    train_loss = 0
    for b, img in enumerate(train_loader):
      optimizer.zero_grad()
      recon_x, mu, logvar = model(img)
      loss = vae_loss(recon_x, img, mu, logvar)
      loss.backward()
      train_loss+=loss.item()
      optimizer.step()
      # print(f'loss at epoch {i} batch {b}: {loss.item()/len(img)}')
  #   print(f"loss at epoch {i}: {train_loss/len(train_loader)}")
  #   with torch.no_grad():
  #     sample = torch.randn(64, 128, 1, 1)
  #     sample = model.decoder(sample)
  #     save_image(sample.view(64, 1, 28, 28), f'./results/{i}.png')


  # torch.save(model.state_dict(), "./models/vae.pt")


