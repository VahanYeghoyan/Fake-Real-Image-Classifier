class TrainConfig:
    def __init__(self, batch_size, learning_rate, epoch, training_data_path,
                 display_step, checkpoint_step, summary_step, max_to_keep=None):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.training_data_path = training_data_path
        self.display_step = display_step
        self.checkpoint_step = checkpoint_step
        self.summary_step = summary_step
        self.max_to_keep = max_to_keep


class ValConfig:
    def __init__(self, batch_size, val_data_path, validation_step):
        self.batch_size = batch_size
        self.val_data_path = val_data_path
        self.validation_step = validation_step


class ModelConfig:
    def __init__(self, name, image_height, image_width, image_channels):
        self.name = name
        self.image_height = image_height
        self.image_width = image_width
        self.image_channels = image_channels


class OptimizerConfig:
    def __init__(self, train_config: TrainConfig, val_config: ValConfig, model_config: ModelConfig):
        self.train_config = train_config
        self.model_config = model_config
        self.val_config = val_config
