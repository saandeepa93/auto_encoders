import os
import torch
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

  trans = transforms.Compose(
    [transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize((0, 0, 0), (1, 1, 1))]
  )
  hourglass = Hourglass(in_channel, out_channel)

  for file in os.listdir(input_path):
    img = cv2.imread(os.path.join(input_path, file))
    img = trans(img).unsqueeze(0).type(torch.float)
    out = hourglass(img)
    # show(out.squeeze().permute(1, 2, 0).detach())


