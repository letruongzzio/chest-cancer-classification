from pathlib import Path
from Chest_Cancer_Classification.constants import *
from Chest_Cancer_Classification.entity.config_entity import DataIngestionConfig
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
