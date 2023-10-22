import numpy as np
import tensorflow as tf
import os
import glob
from network import Network
from config import OptimizerConfig


class optimaizer:
    def __init__(self, config: OptimizerConfig, restore_ckpt_num=0):
        self.config = config
        self.restore_ckpt_num = restore_ckpt_num

        self.ckpt_file_name = 'ckpt'

        self.net = Network(config.model_config)
        # self.opt = tf.optimizers.Adam(learning_rate=self.config.train_config.learning_rate)
        self.train_config = config.train_config
        self.val_config = config.val_config
        self.model_config = config.model_config

        self.summary_dir = os.path.join("models", self.net.name, "summaries")
        self.checkpoint_dir = os.path.join("models", self.net.name, "checkpoints")

        self.train_paths = glob.glob(os.path.join(config.train_config.training_data_path, r"**\*.jpg"), recursive=True)
        self.val_paths = glob.glob(os.path.join(config.val_config.val_data_path, r"**\*.jpg"), recursive=True)
        np.random.shuffle(self.train_paths)
        np.random.shuffle(self.val_paths)
        self.train_model_setup()

    def get_checkpoint_prefix(self):
        checkpoint_prefix = os.path.join(self.checkpoint_dir, self.ckpt_file_name)
        return checkpoint_prefix


    def ckpt_setup(self):
        print("------ckpt_setup------")
        if not self.restore_ckpt_num is None and self.restore_ckpt_num >= 1:
            ckpt_path = self.get_checkpoint_prefix() + '-' + str(self.restore_ckpt_num)

        else:
            ckpt_path = tf.train.latest_checkpoint(self.checkpoint_dir)

        self.checkpoint = tf.train.Checkpoint(optimaizer=self.opt, net=self.net.get_opt_var_dict())
        self.ckpt_manager = tf.train.CheckpointManager(self.checkpoint,
                                                       directory=self.checkpoint_dir,
                                                       max_to_keep=self.train_config.max_to_keep,
                                                       step_counter=self.opt.iterations)

        if ckpt_path is not None:
            print('[*] Restoring ckpt_path = ' + ckpt_path)
            try:
                status = self.checkpoint.restore(ckpt_path)

            except tf.errors.OpError:
                print('problem with checkpoint')
                exit()

    def save_ckpt(self):
        print('[*] saving checkpoint')
        self.ckpt_manager.save()

    @tf.function
    def tf_read_img_and_resize(self, path):


        label = tf.strings.split(path, "\\")[-2]
        label = tf.strings.to_number(label)
        one_hot_label = tf.stack([label, 1 - label])

        img = tf.io.read_file(path)
        img = tf.io.decode_jpeg(img)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [self.model_config.image_height, self.model_config.image_width])
        return img, one_hot_label

    def train_model_setup(self):
        self.opt = tf.optimizers.Adam(learning_rate=self.train_config.learning_rate)

        train_data = tf.data.Dataset.from_tensor_slices(self.train_paths)
        train_data = train_data.map(self.tf_read_img_and_resize, num_parallel_calls=8)
        train_data = train_data.repeat(self.config.train_config.epoch)
        train_data = train_data.shuffle(buffer_size=800, reshuffle_each_iteration=True)
        self.train_data = train_data.batch(self.config.train_config.batch_size)

        val_data = tf.data.Dataset.from_tensor_slices(self.val_paths)
        val_data = val_data.map(self.tf_read_img_and_resize, num_parallel_calls=8)
        val_data = val_data.repeat(self.config.train_config.epoch)
        val_data = val_data.shuffle(buffer_size=800, reshuffle_each_iteration=True)
        self.val_data = val_data.batch(self.config.val_config.batch_size)
        self.val_data = iter(self.val_data)

        self.ckpt_setup()
        self.summary_setup()



    def summary_setup(self):
        self.train_writer = tf.summary.create_file_writer(os.path.join(self.summary_dir, "train"))
        self.val_writer = tf.summary.create_file_writer(os.path.join(self.summary_dir, "val"))

    @tf.function
    def write_train_summary(self, loss_val):
        print("[*] Tracing write_summary")

        with self.train_writer.as_default():
            tf.summary.scalar("loss", loss_val, step=self.opt.iterations)

    @tf.function
    def write_val_summary(self, loss_val):
        print("[*] Tracing write_summary")

        with self.val_writer.as_default():
            tf.summary.scalar("loss", loss_val, step=self.opt.iterations)


    def train_model(self):
        print("[*] Training model ...")


        for train_pair in self.train_data:
            train_inp, train_lab = train_pair


            self.opt.minimize(lambda: self.net.network_and_loss_call(train_inp, train_lab),
                              var_list=self.net.get_opt_var_list())


            if tf.greater(self.opt.iterations,1) \
                    and tf.equal(tf.math.floormod(self.opt.iterations, self.train_config.display_step), 0):
                train_loss_value = self.net.network_and_loss_call(train_inp, train_lab)

                tf.print('step =', self.opt.iterations, ", loss = ", train_loss_value, sep=" ")

            if tf.greater(self.opt.iterations, 1) \
                    and tf.equal(tf.math.floormod(self.opt.iterations, self.train_config.summary_step), 0):
                train_loss_value = self.net.network_and_loss_call(train_inp, train_lab)
                self.write_train_summary(train_loss_value)

            if tf.greater(self.opt.iterations, 1) \
                    and tf.equal(tf.math.floormod(self.opt.iterations, self.train_config.checkpoint_step), 0):
                self.save_ckpt()

            if tf.greater(self.opt.iterations, 1) \
                    and tf.equal(tf.math.floormod(self.opt.iterations, self.val_config.validation_step), 0):

                val_inp, val_lab = next(self.val_data)
                val_loss_value = self.net.network_and_loss_call(val_inp, val_lab)
                self.write_val_summary(val_loss_value)
                tf.print("val Loss", val_loss_value, sep=" ")
