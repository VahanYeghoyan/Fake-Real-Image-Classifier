import tensorflow as tf
import numpy as np
from network import Network
from config import ModelConfig

checkpoint_path = "C:\\Users\\vahan.yeghoyan\\Desktop\\projects\\fake_real\\code\\models\\GANs_val\\checkpoints"

model_config = ModelConfig(name='GANs_val', image_height=224, image_width=224, image_channels=3)


net = Network(model_config)



def preprocess_input(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    img = tf.image.resize(img, [224, 224])

    return img




for i in range(10):
    def inference(image_path):
        preprocessed_input = preprocess_input(image_path)

        checkpoint = tf.train.Checkpoint(optimizer=net.get_opt_var_dict())
        manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_path, max_to_keep=1)

        if manager.latest_checkpoint:
            checkpoint.restore(manager.latest_checkpoint).expect_partial()
            print("Restored checkpoint from:", manager.latest_checkpoint)
        else:
            print("Checkpoint not found. Make sure to save checkpoints during training.")

        predictions = net.network(tf.convert_to_tensor([preprocessed_input], dtype=tf.float32))


        return predictions





if __name__ == "__main__":


    image_path = "C:\\Users\\vahan.yeghoyan\\Desktop\\projects\\fake_real\\dataset\\train\\1\\00015.jpg"

    result = inference(image_path)
    if result[0][0] > 0.5:
        print('real')
    else:
        print('fake')

    print("Inference result:")
    print(result)