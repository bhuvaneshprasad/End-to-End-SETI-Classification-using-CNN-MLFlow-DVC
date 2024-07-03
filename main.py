from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelPipeline
from cnnClassifier import logger

# STAGE_NAME = "Data Ingestion Stage"

# try:
#     logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
#     obj = DataIngestionPipeline()
#     obj.main()
#     logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<")
# except Exception as e:
#     logger.exception(e)
#     raise e

STAGE_NAME = "Prepare Base Model"

try:
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    obj = PrepareBaseModelPipeline()
    obj.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<")
except Exception as e:
    logger.exception(e)
    raise e