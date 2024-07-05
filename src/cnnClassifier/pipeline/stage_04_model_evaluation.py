import os
from venv import logger

import mlflow
from cnnClassifier.components.model_evaluation import Evaluation
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.constants import CONFIG_FILE_PATH
from cnnClassifier.utils.common import read_yaml

STAGE_NAME = "Model Evaluation and MLFlow"

class ModelEvaluationPipeline:
    """
    A class representing a pipeline for evaluating a model.

    Attributes:
        None
    """
    def __init__(self) -> None:
        """
        Initialize the ModelEvaluationPipeline class.
        """
        pass
    
    def main(self):
        """
        Main function to execute the model evaluation pipeline.

        Raises:
            Exception: If there is an error during model evaluation or logging.
        """
        try:
            config = read_yaml(CONFIG_FILE_PATH)
            mlflow.set_tracking_uri(config.model_evaluation.dagshub_uri)
            os.environ["MLFLOW_TRACKING_USERNAME"]=os.getenv('MLFLOW_TRACKING_USERNAME')
            os.environ["MLFLOW_TRACKING_PASSWORD"]=os.getenv('MLFLOW_TRACKING_PASSWORD')
            config = ConfigurationManager()
            eval_config = config.get_evaluation_config()
            eval = Evaluation(eval_config)
            eval.evaluation()
            eval.log_into_mlflow()
        except Exception as e:
            raise e

if __name__ == '__main__':
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<")
    except Exception as e:
        logger.exception(e)
        raise e