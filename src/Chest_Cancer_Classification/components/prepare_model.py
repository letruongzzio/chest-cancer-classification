"""
This module contains the PrepareBaseModel class, which is responsible for preparing a base model for fine-tuning.
It includes methods for loading a pre-trained model, modifying it for a specific task, and saving the updated model.
It uses TensorFlow and Keras for model handling and assumes the use of VGG16 as the base model.
"""

from pathlib import Path
from tensorflow import keras
from Chest_Cancer_Classification.entity.config_entity import PrepareModelConfig


class PrepareModel:
    """
    This class is responsible for preparing the base model for fine-tuning.
    It loads a pre-trained model, modifies it for the specific task, and saves the updated model.
    """
    def __init__(self, config: PrepareModelConfig):
        """
        Initializes the PrepareBaseModel class with the given configuration.
        Args:
            config (PrepareModelConfig): Configuration for preparing the model.
        """
        self.config = config
        self.model = None
        self.full_model = None

    def get_model(self):
        """
        Loads the base model (VGG16) with the specified parameters.
        The model is loaded with the specified input shape, weights, and whether to include the top layers.
        """
        self.model = keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )

        self.save_model(path=self.config.model_path, model=self.model)

    @staticmethod
    def _prepare_full_model(model: keras.Model, classes: int, freeze_all: bool, freeze_till: int, learning_rate: float) -> keras.Model:
        """
        Prepares the full model for fine-tuning using a Sequential model.
        Args:
            model (keras.Model): The base model to be modified.
            classes (int): The number of output classes.
            freeze_all (bool): Whether to freeze all layers of the base model.
            freeze_till (int): The number of layers to freeze from the end of the model.
            learning_rate (float): The learning rate for the optimizer.
        Returns:
            keras.Model: The modified model ready for fine-tuning.
        """
        # Freeze layers based on the provided configuration
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif freeze_till is not None and freeze_till > 0:
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

        # Create a Sequential model and add the base model and new layers
        full_model = keras.Sequential([
            model,
            keras.layers.Flatten(),
            keras.layers.Dense(
                units=classes,
                activation="softmax",
                name="output_layer"
            )
        ])

        # Compile the model with the specified optimizer, loss, and metrics
        full_model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=learning_rate),
            loss=keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        # Print the model summary for debugging purposes
        full_model.summary()
        return full_model

    def update_model(self):
        """
        Updates the base model by adding new layers and compiling it for fine-tuning.
        The model is saved after updating.
        """
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: keras.Model):
        """
        Saves the model to the specified path.
        Args:
            path (Path): The path where the model will be saved.
            model (keras.Model): The model to be saved.
        """
        model.save(path)
