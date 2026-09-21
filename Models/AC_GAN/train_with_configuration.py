import argparse
from pathlib import Path

import yaml

from genome_ac_gan_training import train_genome_ac_model

DEFAULT_CONFIGURATION_PATH = 'configurations/train_continental_model.yaml'


def load_yaml_to_dict(path):
    with open(path, 'r') as f:
        yaml_dict = yaml.safe_load(f)
        return yaml_dict


def main(path):
    configuration = load_yaml_to_dict(path)
    print(f"Loading AC-GAN configuration: {Path(path).resolve()}")
    print("YAML configuration values:")
    for key, value in configuration.items():
        print(f"  {key}: {value}")
    train_genome_ac_model(**configuration)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train with Yaml configuration')
    parser.add_argument('--path', type=str, default=DEFAULT_CONFIGURATION_PATH, help='path to yaml file configuration')
    args = parser.parse_args()
    main(path=args.path)
