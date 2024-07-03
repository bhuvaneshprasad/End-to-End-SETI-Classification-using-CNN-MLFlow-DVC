import os
from pathlib import Path
from urllib.parse import urlparse
import mlflow
import numpy as np
import tensorflow as tf
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import save_json
from mlflow.models.signature import ModelSignature
from mlflow.types import Schema, TensorSpec


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    
    def evaluation(self):
        val_data = os.path.join(self.config.training_data_path, "seti-sub\\seti_signals_sub\\valid")
        val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
                                val_data,
                                shuffle=True,
                                image_size = (self.config.params_image_size[0], self.config.params_image_size[0]),
                                batch_size = self.config.params_batch_size
                            )
        val = val_dataset.cache().shuffle(100).prefetch(buffer_size=tf.data.AUTOTUNE)
        
        self.model = self.load_model(self.config.path_of_model)
        self.score = self.model.evaluate(val)
        self.save_score()

    def save_score(self):
        scores = {'loss': self.score[0], 'accuracy': self.score[1]}
        save_json(path=Path("scores.json"), data=scores)
    
    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )
            if tracking_url_type_store != "file":
                # Define the input and output schema
                input_schema = Schema([TensorSpec(np.dtype("float32"), (-1, self.config.params_image_size[0], self.config.params_image_size[1], self.config.params_image_size[2]))])
                output_schema = Schema([TensorSpec(np.dtype("float32"), (-1, self.config.params_classes))])

                # Create the model signature
                signature = ModelSignature(inputs=input_schema, outputs=output_schema)
                mlflow.keras.log_model(self.model, "model", registered_model_name="InceptionV3", signature=signature)
            else:
                mlflow.keras.log_model(self.model, "model")