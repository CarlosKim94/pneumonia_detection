import numpy as np
import os
import onnxruntime as ort
from keras_image_helper import create_preprocessor
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, HttpUrl
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
    url: HttpUrl

# HTML frontend for testing
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <body>
            <h2>Pneumonia Detection (URL)</h2>
            <form action="/predict" method="post">
                <input type="text" name="url" placeholder="Enter image URL" size="60" required>
                <br><br>
                <button type="submit">Predict</button>
            </form>
        </body>
    </html>
    """

def github_raw_url(url):
    if "github.com" in url and "/blob/" in url:
        url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    return url

@app.post("/predict")
def predict(url: str = Form(...)):
    # Preprocess image directly from URL
    url = github_raw_url(url)
    X = preprocessor.from_url(url)

    # ONNX inference
    logits = session.run([output_name], {input_name: X})[0][0]

    # Sigmoid probability
    prob = float(1 / (1 + np.exp(-logits.item())))
    label = "PNEUMONIA" if prob > 0.5 else "NORMAL"

    return {"prediction": label, "probability": round(prob, 4)}

@app.get("/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)