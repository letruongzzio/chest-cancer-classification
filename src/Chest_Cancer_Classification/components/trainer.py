from pathlib import Path
from tensorflow import keras
from src.Chest_Cancer_Classification.config.configuration import ConfigurationManager
from Chest_Cancer_Classification.entity.config_entity import TrainingConfig


class Trainer:
    """
    This class is responsible for training the model using the Keras library.
    It initializes the model, sets up data generators for training and validation,
    and provides methods to train the model and save it to a specified path.
    """
    def __init__(self, config: ConfigurationManager):
        """
        Initializes the Training class with the given configuration.
        Args:
            config (TrainingConfig): Configuration object containing training parameters.
        """
        self.config = config
        self.prepare_model_config = config.get_prepare_model_config()
        self.training_config = config.get_training_config()
        self.model = None
        self.train_generator = None
        self.valid_generator = None
        self.steps_per_epoch = None
        self.validation_steps = None
        self.get_model()
        self.train_valid_generator()

    
    def get_model(self):
        """
        Loads the base model from the specified path and compiles it.
        """
        self.model = keras.models.load_model(
            self.prepare_model_config.updated_model_path
        )

    def train_valid_generator(self):
        """
        Sets up the data generators for training and validation.
        It uses the ImageDataGenerator class from Keras to create data generators
        that can augment the training data and normalize the pixel values.
        The training data is split into training and validation sets based on the
        specified validation split in the configuration.
        """
        # Define the data generator parameters
        datagenerator_kwargs = dict(
            rescale=1./255,         # Normalize pixel values to [0, 1]
            validation_split=0.20   # Split the data into training and validation sets
        )

        # Define the data flow parameters
        dataflow_kwargs = dict(
            target_size=self.training_config.params_image_size[:-1],
            batch_size=self.training_config.params_batch_size,
            interpolation="bilinear"
        )

        # Create a validation generator
        valid_datagenerator = keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )
        
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.training_config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        if self.training_config.params_is_augmentation:
            train_datagenerator = keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.training_config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

    @staticmethod
    def save_model(path: Path, model: keras.Model):
        """
        Saves the trained model to the specified path.
        Args:
            path (Path): The path where the model will be saved.
            model (keras.Model): The trained Keras model to be saved.
        """
        model.save(path)
    
    def train(self):
        """
        Trains the model using the training and validation data generators.
        It sets the number of steps per epoch and validation steps based on the
        number of samples in the training and validation data.
        The model is trained for the specified number of epochs and the trained
        model is saved to the specified path.
        """
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        self.model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=self.prepare_model_config.params_learning_rate),
            loss=keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        self.model.fit(
            self.train_generator,
            epochs=self.training_config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator
        )

        self.save_model(
            path=self.training_config.trained_model_path,
            model=self.model
        )
