from config import *
from optimaizer import optimaizer


def main():
    Train_Config = TrainConfig(batch_size=50,
                                learning_rate=0.001,
                                epoch=10,
                                training_data_path="C:\\Users\\vahan.yeghoyan\\Desktop\\projects\\fake_real\\dataset\\train",
                                display_step=10,
                                checkpoint_step=50,
                                summary_step=50)
    Val_config = ValConfig(batch_size=50, val_data_path="C:\\Users\\vahan.yeghoyan\\Desktop\\projects\\fake_real\\dataset\\test", validation_step=10)

    Model_config = ModelConfig(name='GANs_val', image_height=224, image_width=224, image_channels=3)

    Optimaizer_config = OptimizerConfig(Train_Config, Val_config, Model_config)

    Optimaizer = optimaizer(Optimaizer_config)

    Optimaizer.train_model()


if __name__ == "__main__":
    main()




