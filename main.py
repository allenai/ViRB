from models.ResNet50Encoder import ResNet50Encoder
from VTABRunner import VTABRunner
import sys


def main():
    runner = VTABRunner(
        encoder=ResNet50Encoder(weights=sys.argv[2]),
        run_name=sys.argv[1],
        train_encoder=True,
    )
    runner.run()


if __name__ == '__main__':
    main()
