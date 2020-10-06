import cv2
import matplotlib.pyplot as plt
import yaml
import torch
from torch.nn import functional as F


def show(img):
    plt.imshow((img * 255).type(torch.uint8))
    plt.show()

def imshow(img):
  cv2.imshow("img", img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

def get_config(config_path):
  with open(config_path) as file:
    configs = yaml.load(file, Loader = yaml.FullLoader)
  return configs

def vae_loss(recon_x, x,  mu, logvar):
  BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
  KLD = -0.5 * torch.sum(1+logvar-mu.pow(2)-logvar.exp())
  return BCE+KLD


