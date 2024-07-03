import os
from pathlib import Path
import tensorflow as tf
from cnnClassifier.entity.config_entity import ModelTrainingConfig


class ModelTraining:
    def __init__(self, config: ModelTrainingConfig) -> None:
        self.config = config
    
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )
    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    
    def train(self):
        train_data = os.path.join(self.config.training_data_path, "train")
        val_data = os.path.join(self.config.training_data_path, "valid")
        
        train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
                                train_data,
                                shuffle=True,
                                image_size = (self.config.params_image_size[0], self.config.params_image_size[0]),
                                batch_size = self.config.params_batch_size
                            )
        
        val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
                                val_data,
                                shuffle=True,
                                image_size = (self.config.params_image_size[0], self.config.params_image_size[0]),
                                batch_size = self.config.params_batch_size
                            )
        
        train = train_dataset.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
        val = val_dataset.cache().shuffle(100).prefetch(buffer_size=tf.data.AUTOTUNE)
        
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

        self.model.fit(
            train,
            epochs=self.config.params_epochs,
            batch_size=self.config.params_batch_size,
            verbose=1,
            validation_data=val,
            callbacks=[early_stopping]
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )