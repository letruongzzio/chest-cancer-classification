from src.Chest_Cancer_Classification import logger
from src.Chest_Cancer_Classification.config.configuration import ConfigurationManager
from src.Chest_Cancer_Classification.pipeline.data_ingestion_pipeline import DataIngestionPipeline
from src.Chest_Cancer_Classification.pipeline.prepare_model_pipeline import PrepareModelTrainingPipeline
from src.Chest_Cancer_Classification.constants import *

config = ConfigurationManager()

if __name__ == "__main__":
    try:
        logger.info(f"{'>>'*20} {'Pipeline Execution Started'} {'<<'*20}")

        logger.info(f"{'>>'*20} STAGE 1: Data Ingestion {'<<'*20}")
        data_ingestion_pipeline = DataIngestionPipeline(config=config)
        data_ingestion_pipeline.main()
        logger.info("Data Ingestion Pipeline completed successfully.")

        logger.info(f"{'>>'*20} STAGE 2: Prepare Model {'<<'*20}")
        prepare_model_pipeline = PrepareModelTrainingPipeline(config=config)
        prepare_model_pipeline.main()
        logger.info("Prepare Model Pipeline completed successfully.")

        logger.info(f"{'>>'*20} {'Pipeline Execution Completed'} {'<<'*20}")
    except Exception as e:
        logger.exception(f"Exception occurred during pipeline execution: {e}")
        raise e
