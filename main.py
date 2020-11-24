from models.ResNet50Encoder import ResNet50Encoder
from VTABRunner import VTABRunner


def main():
    runner = VTABRunner(
        run_name="SWAV_800",
        encoder_class=ResNet50Encoder,
        encoder_args={"weights": "pretrained_weights/SWAV_800.pt"},
        train_encoder=False,
    )
    runner.run()


if __name__ == '__main__':
    main()
