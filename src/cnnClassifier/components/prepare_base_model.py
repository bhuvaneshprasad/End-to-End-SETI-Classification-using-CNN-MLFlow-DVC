from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, applications

from cnnClassifier.entity.config_entity import PrepareBaseModelConfig
from cnnClassifier.utils.common import create_directories

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig) -> None:
        self.config = config
    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
    
    def get_base_model(self):
        self.model = applications.InceptionV3(
                        include_top=self.config.params_include_top,
                        weights=self.config.params_weights,
                        input_shape=self.config.params_image_size,
                    )
        
        create_directories([self.config.root_dir])
        
        self.save_model(path=self.config.base_model_path, model=self.model)
    
    @staticmethod
    def _prepare_full_model(model, classes, learning_rate, image_size_int, input_shape, trainable=False, augmentation=False):
        if trainable:
            model.trainable = True
        else:
            model.trainable = False
        
        full_model = tf.keras.Sequential()
        
        if augmentation:
            full_model.add(layers.Resizing(image_size_int, image_size_int))
            full_model.add(layers.Rescaling(1.0/image_size_int))
        
        full_model.add(model)
        full_model.add(layers.GlobalAveragePooling2D())
        full_model.add(layers.Flatten())
        full_model.add(layers.Dense(units=208, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)))
        full_model.add(layers.Dropout(0.3))
        full_model.add(layers.Dense(units=208, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)))
        full_model.add(layers.Dropout(0.3))
        full_model.add(layers.Dense(units=208, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)))
        full_model.add(layers.Dropout(0.2))
        full_model.add(layers.Dense(units=classes, activation='softmax'))

        full_model.build(input_shape = input_shape)

        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy']
        )
        
        full_model.summary()
        return full_model
    
    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            learning_rate=self.config.params_learning_rate,
            image_size_int=self.config.params_image_size[0],
            input_shape=[self.config.params_batch_size] + self.config.params_image_size,
            trainable=True,
            augmentation=self.config.params_augmentation
        )
        
        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)