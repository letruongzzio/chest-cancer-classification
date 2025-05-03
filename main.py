from src.Chest_Cancer_Classification import logger
from src.Chest_Cancer_Classification.pipeline.data_ingestion_pipeline import run_data_ingestion_pipeline
from src.Chest_Cancer_Classification.constants import *

if __name__ == "__main__":
    try:
        logger.info(f"{'>>'*20} {'Pipeline Execution Started'} {'<<'*20}")
        run_data_ingestion_pipeline()
        logger.info(f"{'>>'*20} {'Pipeline Execution Completed'} {'<<'*20}")
    except Exception as e:
        logger.exception(f"Exception occurred during pipeline execution: {e}")
        raise e
