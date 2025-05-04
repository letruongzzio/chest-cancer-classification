"""
This module contains the DataIngestionPipeline class, which is responsible for managing the data ingestion pipeline.
It orchestrates the downloading and extraction of the dataset.
It uses the ConfigurationManager to get the configuration settings and the DataIngestion component to perform the actual data ingestion.
"""

from Chest_Cancer_Classification import logger
from Chest_Cancer_Classification.config.configuration import ConfigurationManager
from Chest_Cancer_Classification.components.data_ingestion import DataIngestion

STAGE_NAME = "Stage 1: Data Ingestion"

class DataIngestionPipeline:
    """
    This class is responsible for managing the data ingestion pipeline.
    It orchestrates the downloading and extraction of the dataset.
    """
    def __init__(self, config: ConfigurationManager):
        self.config = config

    def main(self):
        """
        The main method of the DataIngestionPipeline class.
        It initializes the configuration manager and the data ingestion component.
        It downloads the dataset and extracts it.
        """
        try:
            data_ingestion_config = self.config.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)
            data_ingestion.download_file()
            data_ingestion.extract_zip_file()
        except Exception as e:
            raise e

