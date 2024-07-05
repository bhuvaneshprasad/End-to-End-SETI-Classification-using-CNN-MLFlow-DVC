from pathlib import Path
import tensorflow as tf

from cnnClassifier.entity.config_entity import PrepareBaseModelConfig
from cnnClassifier.utils.common import create_directories

class PrepareBaseModel:
    """
    A class to prepare and save the base model for training, and to update the base model 
    by adding custom layers and compiling it.

    Attributes:
        config (PrepareBaseModelConfig): Configuration for preparing the base model.
    """
    def __init__(self, config: PrepareBaseModelConfig) -> None:
        """
        Initialize the PrepareBaseModel class with the given configuration.

        Args:
            config (PrepareBaseModelConfig): Configuration for preparing the base model.
        """
        self.config = config
    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """
        Save the model to the specified path.

        Args:
            path (Path): Path to save the model.
            model (tf.keras.Model): Model to be saved.
        """
        model.save(path)
    
    def get_base_model(self):
        """
        Load the base model (InceptionV3) with the specified configuration and save it.
        """
        self.model = tf.keras.applications.inception_v3.InceptionV3(
                        include_top=self.config.params_include_top,
                        weights=self.config.params_weights,
                        input_shape=self.config.params_image_size,
                    )
        
        create_directories([self.config.root_dir])
        
        self.save_model(path=self.config.base_model_path, model=self.model)
    
    @staticmethod
    def _prepare_full_model(model, classes, learning_rate, trainable=False):
        """
        Prepare the full model by adding custom layers and compiling it.

        Args:
            model (tf.keras.Model): Base model to be updated.
            classes (int): Number of output classes.
            learning_rate (float): Learning rate for the optimizer.
            trainable (bool): Whether to make the base model trainable.

        Returns:
            tf.keras.Model: Full model with custom layers added and compiled.
        """
        if trainable:
            model.trainable = True
        else:
            model.trainable = False
        
        full_model = tf.keras.layers.Flatten()(model.output)
        full_model = tf.keras.layers.Dense(448, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01))(full_model)
        full_model = tf.keras.layers.Dropout(0.2)(full_model)
        full_model = tf.keras.layers.Dense(classes, activation='softmax')(full_model)

        full_model = tf.keras.Model(model.input, full_model)

        full_model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                    metrics=['accuracy']
                )
        
        full_model.summary()
        return full_model
    
    def update_base_model(self):
        """
        Update the base model by adding custom layers and compiling it, then save the updated model.
        """
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            learning_rate=self.config.params_learning_rate,
            trainable=True
        )
        
        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)