import os
import torch
import torchvision
from torch import nn, optim
from torchvision import transforms
from PIL import Image
import cv2
from sys import exit as e

from modules.hourglass import Hourglass
from modules.util import imshow, show


def train_data(configs):
  input_path = configs["path"]["input"]
  im_size = configs["params"]["img_size"]
  in_channel = configs["params"]["in_channels"]
  out_channel = configs["params"]["out_channels"]
  encoder_feats = configs["params"]["encoder"]
  decoder_feats = configs["params"]["decoder"]
  lr = float(configs["params"]["lr"])
  epochs = float(configs["params"]["epochs"])

  transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

  train_dataset = torchvision.datasets.MNIST(
    root="~/torch_datasets", train=True, transform=transform, download=True
  )

  train_loader = torch.utils.data.DataLoader(
      train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
  )

  trans = transforms.Compose(
    [transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize((0, 0, 0), (1, 1, 1))]
  )
  hourglass = Hourglass(in_channel, out_channel, encoder_feats, decoder_feats)
  optimizer = optim.Adam(hourglass.parameters(), lr)
  criterion = nn.MSELoss()

  for i in range(int(epochs)):
    total_loss = 0
    for imgs, _ in train_loader:
      optimizer.zero_grad()
      out = hourglass(imgs)
      loss = criterion(out, imgs)
      loss.backward()
      optimizer.step()
      total_loss+=loss.item()
    print(f"average loss after {i+1} epochs: {total_loss/len(train_loader)}")
  torch.save(hourglass.state_dict(), "./models/model.pt")


