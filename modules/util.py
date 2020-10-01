import cv2
import matplotlib.pyplot as plt
import yaml
import torch


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
