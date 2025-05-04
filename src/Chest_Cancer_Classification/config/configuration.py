"""
This module contains the ConfigurationManager class, which is responsible for managing the configuration of the project.
It reads the configuration and parameters from YAML files and provides methods to access them.
It uses the `read_yaml` function to read the YAML files and the `create_directories` function to create necessary directories.
It also defines the `ConfigurationManager` class, which provides methods to get the data ingestion and model preparation configurations.
"""

import os
from pathlib import Path
from Chest_Cancer_Classification.constants import *
from Chest_Cancer_Classification.entity.config_entity import *
from Chest_Cancer_Classification.utils.common import read_yaml, create_directories

class ConfigurationManager:
    """
    This class is responsible for managing the configuration of the project.
    It reads the configuration and parameters from YAML files and provides methods to access them.
    
    Attributes:
        config (dict): Configuration settings.
        params (dict): Parameters settings.
    """
    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(str(config_filepath))
        self.params = read_yaml(str(params_filepath))
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        This method is responsible for setting up the data ingestion configuration.
        It creates the necessary directories and prepares the configuration for data ingestion.

        Returns:
            DataIngestionConfig: The data ingestion configuration object.
        """
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        
        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=config.source_URL,
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir)
        )
        return data_ingestion_config

    def get_prepare_model_config(self) -> PrepareModelConfig:
        """
        This method is responsible for setting up the model preparation configuration.
        It creates the necessary directories and prepares the configuration for model preparation.

        Returns:
            PrepareModelConfig: The model preparation configuration object.
        """
        config = self.config.prepare_model
        create_directories([config.root_dir])
        
        prepare_model_config = PrepareModelConfig(
            root_dir=Path(config.root_dir),
            model_path=Path(config.model_path),
            updated_model_path=Path(config.updated_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES
        )
        return prepare_model_config
    
    def get_training_config(self) -> TrainingConfig:
        """
        This method is responsible for setting up the training configuration.
        It creates the necessary directories and prepares the configuration for training.
        Returns:
            TrainingConfig: The training configuration object.
        """
        training = self.config.training
        prepare_model = self.config.prepare_model
        params = self.params
        training_data = os.path.join(self.config.data_ingestion.unzip_dir, "Chest-CT-Scan-data")
        create_directories([Path(training.root_dir)])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_model.updated_model_path),
            training_data=Path(training_data),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE
        )

        return training_config
