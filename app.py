import os
import shutil
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from cnnClassifier.pipeline.prediction import PredictionPipeline

app = FastAPI(swagger_ui_parameters={"defaultModelsExpandDepth": -1})

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def classification(file: UploadFile):
    temp_file_path = f"temp_{file.filename}"
    
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        predictor = PredictionPipeline(temp_file_path)
        prediction = predictor.predict()
    finally:
        os.remove(temp_file_path)

    return {'prediction' : prediction}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=7384)