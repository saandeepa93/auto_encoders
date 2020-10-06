import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from sys import exit as e


class Dset(Dataset):
  def __init__(self, root_dir, img_size, ext):
    super(Dset, self).__init__()
    self.root_dir = root_dir
    self.ext = ext
    self.isize = img_size
    self.all_files = [f for f in os.listdir(self.root_dir)]

    self.transform = transforms.Compose(
    [transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0), (1))]
    )

  def __len__(self):
    return len(self.all_files)


  def __getitem__(self, idx):
    img = Image.open(os.path.join(self.root_dir, self.all_files[idx]))
    img = self.transform(img)
    return img



