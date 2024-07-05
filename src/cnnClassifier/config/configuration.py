import os
from cnnClassifier.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import (
                                                DataIngestionConfig,
                                                EvaluationConfig,
                                                ModelTrainingConfig,
                                                PrepareBaseModelConfig)
from pathlib import Path

class ConfigurationManager:
    """
    A class to manage the configuration settings for the CNN classifier.

    Attributes:
        config_filepath (str): Path to the configuration file.
        params_filepath (str): Path to the parameters file.
    """
    def __init__(self, config_filepath = CONFIG_FILE_PATH, params_filepath = PARAMS_FILE_PATH) -> None:
        """
        Initialize the ConfigurationManager with paths to the configuration and parameters files.

        Args:
            config_filepath (str): Path to the configuration file.
            params_filepath (str): Path to the parameters file.
        """
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        
        create_directories([self.config.artifacts_root])
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Get the Data Ingestion configuration.

        Returns:
            DataIngestionConfig: Data ingestion configuration settings.
        """
        config = self.config.data_ingestion
        
        create_directories([config.root_dir])
        
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_url=config.source_url,
            data_dir=config.data_dir
        )
        
        return data_ingestion_config
    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        """
        Get the Prepare Base Model configuration.

        Returns:
            PrepareBaseModelConfig: Prepare base model configuration settings.
        """
        config = self.config.prepare_base_model
        
        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_augmentation=self.params.AUGMENTATION,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES
        )
        
        return prepare_base_model_config
    
    def get_model_training_config(self) -> ModelTrainingConfig:
        """
        Get the Model Training configuration.

        Returns:
            ModelTrainingConfig: Model training configuration settings.
        """
        model_training = self.config.model_training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        training_data = os.path.join(self.config.data_ingestion.data_dir, "seti-sub\\seti_signals_sub")
        
        create_directories([model_training.root_dir])
        
        model_training_config = ModelTrainingConfig(
            root_dir=Path(model_training.root_dir),
            trained_model_path=Path(model_training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data_path=Path(training_data),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_image_size=params.IMAGE_SIZE,
            params_is_augmentation=params.AUGMENTATION
        )
        
        return model_training_config
    
    def get_evaluation_config(self) -> EvaluationConfig:
        """
        Get the Evaluation configuration.

        Returns:
            EvaluationConfig: Evaluation configuration settings.
        """
        eval_config = EvaluationConfig(
            path_of_model=self.config.model_training.trained_model_path,
            training_data_path=self.config.data_ingestion.data_dir,
            mlflow_uri=self.config.model_evaluation.dagshub_uri,
            all_params=self.params,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE,
            params_classes=self.params.CLASSES
        )
        return eval_config