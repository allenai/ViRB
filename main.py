from models.ResNet50Encoder import ResNet50Encoder
from VTABRunner import VTABRunner


def main():
    runner = VTABRunner(
        encoder=ResNet50Encoder(weights="pretrained_weights/SWAV_800.pt"),
        run_name="SWAV_800",
        train_encoder=False,
    )
    runner.run()


if __name__ == '__main__':
    main()
