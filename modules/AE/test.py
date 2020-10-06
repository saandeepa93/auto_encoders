import torch
import torchvision
from torchvision import transforms
from sys import exit as e

from modules.hourglass import Hourglass
from modules.util import show

def test_data(configs):
  in_channel = configs["params"]["in_channels"]
  out_channel = configs["params"]["out_channels"]
  encoder_feats = configs["params"]["encoder"]
  decoder_feats = configs["params"]["decoder"]
  model_path = configs["path"]["model"]

  hourglass = Hourglass(in_channel, out_channel, encoder_feats, decoder_feats)
  hourglass.load_state_dict(torch.load(model_path))
  hourglass.eval()

  transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
  test_dataset = torchvision.datasets.MNIST(
      root="~/torch_datasets", train=False, transform=transform, download=True
  )
  test_loader = torch.utils.data.DataLoader(
      test_dataset, batch_size=32, shuffle=False, num_workers=4
  )

  for img,_ in test_loader:
    print(img.size())
    out = hourglass(img)
    print(out.size())
    for i in range(out.size(0)):
      show(img[i].squeeze().detach())
      show(out[i].squeeze().detach())
    e()


