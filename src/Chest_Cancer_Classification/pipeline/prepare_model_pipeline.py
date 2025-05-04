"""
This module defines the PrepareModelTrainingPipeline class, which is responsible for preparing the model for training.
It loads a pre-trained model, modifies it for the specific task, and saves the updated model.
"""

from src.Chest_Cancer_Classification.config.configuration import ConfigurationManager
from src.Chest_Cancer_Classification.components.prepare_model import PrepareModel

class PrepareModelTrainingPipeline:
    """
    This class is responsible for preparing the model for training.
    It loads a pre-trained model, modifies it for the specific task, and saves the updated model.
    """
    def __init__(self, config: ConfigurationManager):
        """
        Initializes the PrepareModelTrainingPipeline class.
        This class is responsible for preparing the model for training.
        """
        self.config = config

    def main(self):
        """
        Main method to execute the model preparation pipeline.
        It loads the base model, modifies it for the specific task, and saves the updated model.
        """
        prepare_model_config = self.config.get_prepare_model_config()
        prepare_model = PrepareModel(config=prepare_model_config)
        prepare_model.get_model()
        prepare_model.update_model()
