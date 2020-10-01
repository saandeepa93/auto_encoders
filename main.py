import click

from modules.util import get_config
from modules.train import train_data


@click.command()
@click.option('--config', help='path of config file')
def run(config):
  configs = get_config(config)
  train_data(configs)

@click.group()
def main():
  pass


if __name__ == '__main__':
  main.add_command(run)
  main()