from models.ResNet50Encoder import ResNet50Encoder
from VTABRunner import VTABRunner
import argparse
import yaml


def main():
    experiment_list = parse_args()
    runner = VTABRunner(
        experiments={
            name: ResNet50Encoder(weights=weights)
            for name, weights in experiment_list.items()
        },
        train_encoder=False
    )
    runner.run()


def parse_args():
    parser = argparse.ArgumentParser(description='pyTorch VTAB')
    parser.add_argument('--experiment_list', help='Path to file with config of runs')
    parser.add_argument('--name', help='Name of current run')
    parser.add_argument('--encoder_weights', help='Path to the weights of the encoder to use')
    args = parser.parse_args()
    if args.experiment_list is not None:
        with open(args.experiment_list) as file:
            experiment_list = yaml.load(file, Loader=yaml.FullLoader)
            return experiment_list
    return {args.name: args.encoder_weights}


if __name__ == '__main__':
    main()
