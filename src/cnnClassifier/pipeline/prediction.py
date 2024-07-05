import os
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

load_dotenv()

class PredictionPipeline:
    """
    A class representing a pipeline for making predictions using a pre-trained model.

    Attributes:
        filename (str): The filename of the image to predict.

    Methods:
        predict() -> int:
            Loads a pre-trained model, processes an image, and predicts its class.
    """
    def __init__(self,filename):
        """
        Initialize the PredictionPipeline class.

        Args:
            filename (str): The filename of the image to predict.
        """
        self.filename =filename
    
    def predict(self) -> int:
        """
        Perform prediction on the image specified by the filename.

        Returns:
            int: The predicted class label.
        """
        model = load_model(Path(os.getenv('MODEL_URI')))

        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (256,256))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = np.argmax(model.predict(test_image), axis=1)
        return int(result)