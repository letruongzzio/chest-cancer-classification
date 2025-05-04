"""
This module contains the ModelTrainingPipeline class, which is responsible for training the model.
It initializes the model, sets up data generators for training and validation,
and provides methods to train the model and save it to a specified path.
"""

from src.Chest_Cancer_Classification.config.configuration import ConfigurationManager
from src.Chest_Cancer_Classification.components.trainer import Trainer

class ModelTrainingPipeline:
    """
    This class is responsible for training the model.
    It initializes the model, sets up data generators for training and validation,
    and provides methods to train the model and save it to a specified path.
    """
    def __init__(self, config: ConfigurationManager):
        """
        Initializes the ModelTrainingPipeline class.
        This class is responsible for training the model.
        """
        self.config = config

    def main(self):
        """
        Main method to execute the model training pipeline.
        It loads the model, sets up data generators, and trains the model.
        """
        training = Trainer(config=self.config)
        training.train()
