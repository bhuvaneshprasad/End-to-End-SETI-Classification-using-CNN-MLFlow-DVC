from venv import logger
from cnnClassifier.components.model_training import ModelTraining
from cnnClassifier.config.configuration import ConfigurationManager

STAGE_NAME = "Model Training"

class ModelTrainingPipeline:
    def __init__(self) -> None:
        pass
    
    def main(self):
        try:
            config = ConfigurationManager()
            model_training_config = config.get_model_training_config()
            model_training = ModelTraining(config=model_training_config)
            model_training.get_base_model()
            model_training.train()
        except Exception as e:
            raise e

if __name__ == '__main__':
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<")
    except Exception as e:
        logger.exception(e)
        raise e