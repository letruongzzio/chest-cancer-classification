import os
import zipfile
import gdown
from Chest_Cancer_Classification.constants import *
from Chest_Cancer_Classification import logger
from Chest_Cancer_Classification.entity.config_entity import DataIngestionConfig

class DataIngestion:
    """
    This class is responsible for downloading and extracting the dataset.
    It handles the downloading of the dataset from a specified URL and extracts it to a specified directory.
    """
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self) -> str:
        """
        This method downloads the dataset from the specified URL.
        It creates the directory if it does not exist.
        It uses gdown to download the file from Google Drive.
        """
        try:
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            os.makedirs(self.config.root_dir, exist_ok=True)
            logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            file_id = dataset_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            gdown.download(prefix + file_id, str(zip_download_dir))
            logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")

        except Exception as e:
            logger.error(f"Error occurred while downloading the file: {e}")
            raise e
        
    def extract_zip_file(self):
        """
        This method extracts the downloaded zip file into the specified directory.
        It creates the directory if it does not exist.
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        logger.info(f"Extracting file {self.config.local_data_file} into directory {unzip_path}")
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
