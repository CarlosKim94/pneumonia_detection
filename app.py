import numpy as np
import os
import onnxruntime as ort
from keras_image_helper import create_preprocessor
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

model_name = os.getenv("MODEL_NAME", "model/pneumonia_mobilenet_v2.onnx")

app = FastAPI(title="pneumonia-detection")

def preprocess_pytorch_style(X):
    X = X / 255.0
    
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    
    X = X.transpose(0, 3, 1, 2)
    X = (X - mean) / std
    
    return X.astype(np.float32)

preprocessor = create_preprocessor(
    preprocess_pytorch_style,
    target_size=(224, 224)
)

session = ort.InferenceSession(
    model_name, providers=["CPUExecutionProvider"]
)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Define class labels
classes = ['NORMAL', 'PNEUMONIA']

class PredictRequest(BaseModel):
    image_path: str

class PredictResponse(BaseModel):
    predictions: dict[str, float]
    top_class: str
    top_probability: float

def predict_single(request: PredictRequest):
    X = preprocessor.from_path(request.image_path)
    logits = session.run([output_name], {input_name: X})[0][0]

    probs = 1 / (1 + np.exp(-logits))  # sigmoid
    probs = float(np.squeeze(probs))
    probs_dict = [round(1 - probs, 4), round(probs, 4)]

    return dict(zip(classes, probs_dict))

@app.post("/predict")
def predict(request: PredictRequest) -> PredictResponse:
    predictions = predict_single(request)
    top_class = max(predictions, key=predictions.get)
    top_probability = predictions[top_class]
    return PredictResponse(
        predictions=predictions,
        top_class=top_class,
        top_probability=top_probability
    )

@app.get("/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)