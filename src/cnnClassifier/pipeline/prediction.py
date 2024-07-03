import os
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

load_dotenv()

class PredictionPipeline:
    def __init__(self,filename):
        self.filename =filename
    
    def predict(self) -> int:
        model = load_model(Path(os.getenv('MODEL_URI')))

        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (224,224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = np.argmax(model.predict(test_image), axis=1)
        return result