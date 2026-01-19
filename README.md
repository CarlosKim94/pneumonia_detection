# Pneumonia Detection using Deep Learning

## 📌 Overview
This project implements a deep learning pipeline for binary classification of chest X-ray images to detect pneumonia. The objective of this project is to build an image classifier that can assist in automated screening and decision support for pneumonia detection from chest X-rays. The model is trained and evaluated on the publicly available Chest X-Ray Pneumonia dataset from **Kaggle by Paul Mooney**. The model leverages a pretrained MobileNetV2 architecture, fine-tuned with various hyperparameters such as learning rate and dropout. Results are evaluated using standard metrics such as accuracy, precision, recall, F1-score, confusion matrix, and ROC AUC.

---

## ❓ Problem Description
Pneumonia is a serious respiratory infection that can be life-threatening if not diagnosed and treated promptly. Chest X-ray imaging is one of the most common and accessible diagnostic tools used by clinicians to detect pneumonia. However, manual interpretation of X-ray images is time-consuming and subject to inter-observer variability, particularly in resource-constrained healthcare settings.

This project explores the use of deep learning–based image classification to automatically detect pneumonia from chest X-ray images. A convolutional neural network is trained to distinguish between NORMAL and PNEUMONIA cases. The model is built using PyTorch and leverages transfer learning with a pretrained MobileNetV2 architecture to achieve strong performance while remaining computationally efficient.

---

## 📊 Dataset
The dataset consists of frontal chest radiographs organized into three subsets including train (5216 images), validation (16 images), and test (624 images), with two class labels: NORMAL and PNEUMONIA. Pneumonia cases include both bacterial and viral infections, while normal cases mean healthy.

Dataset Details:
- Classes:
  - NORMAL
  - PNEUMONIA
- Directories:
  - Training: 5,216 images (Imbalanced: 1,341 PNEUMONIA images and 3,875 NORMAL images)
  - Validation: 16 images (8 PNEUMONIA images and 8 NORMAL images)
  - Test: 624 images (234 PNEUMONIA images and 390 NORMAL images)

---
## 📝 Methodology
Chest X-Ray Images ⮕ Data Preprocessing (resize, normalize, augmentation) ⮕ MobileNetV2 (Pretrained Backbone) ⮕ Custom Classifier (Binary Output) ⮕ Training & Validation (Hyperparameter Tuning) ⮕ Final Evaluation (Test Set Metrics) ⮕ Model Export & Deployment (ONNX / API)

### 1. Data Preparation
Chest X-ray images are loaded from the dataset and split into training, validation, and test sets.

### 2. Preprocessing & Augmentation
Images are resized to 224×224, normalized, and converted to RGB format. Data augmentation is applied only to the training set to improve generalization.

### 3. Model Architecture
A pretrained MobileNetV2 model is used as the feature extractor. The original classifier is replaced with a custom binary classification head.

### 4. Training & Fine-Tuning
Transfer learning is applied by freezing the backbone network and fine-tuning the classifier using binary cross-entropy loss and the Adam optimizer. Learning rate and dropout values are tuned using validation performance.

### 5. Evaluation
The final model is evaluated on a held-out test set using accuracy, precision, recall, F1-score, confusion matrix, and ROC AUC.

### 6. Export & Deployment
The trained model is exported to ONNX format and prepared for deployment as an inference service.

---

## 📊 Results
The final model was evaluated on a test set of 624 chest X-ray images (234 NORMAL, 390 PNEUMONIA) and achieved an overall accuracy of 85%. The model demonstrated strong discriminative performance with an AUC of 0.936.

Recall for PNEUMONIA reached 0.95, indicating high sensitivity and effective detection of infected cases, while recall for NORMAL cases was 0.68. Out of 390 pneumonia cases, 371 were correctly classified, with only 19 false negatives, which is critical in clinical screening scenarios. Although 75 normal cases were misclassified as pneumonia, this trade-off favours patient safety by minimising missed pneumonia diagnoses.

- Key findings:
  - Achieved an overall accuracy of 0.85
  - High sensitivity to pneumonia: Recall for PNEUMONIA is 0.95
  - 371 / 390 pneumonia cases correctly classified with only  19 false negatives
  - Model behavior prioritizes patient safety by minimizing missed pneumonia cases
    ![result](https://github.com/CarlosKim94/credit_risk_prediction/blob/main/EDA/result.png)
  - After hyper parameter tuning, the best performing model is learning_rate = 0.001 and dropout_rate = 0.2, and the model is saved as 'pneumonia_mobilenet_v2.onnx'
  - Fine-tuned MobileNetV2 model ROC_AUC score on test set is 0.936

---

## Repository Structure

```bash
├── EDA
│   ├── correlation.png
│   ├── correlation_heatmap.png
│   ├── loan_intent.png
│   ├── loan_status_distribution.png
│   ├── prediction_test.png
│   └── result.png
├── data
│   └── credit_risk_dataset.csv
├── model
│   ├── model_depth_10_estimator_10_0.839.bin
│   ├── model_depth_10_estimator_20_0.845.bin
│   ├── model_depth_10_estimator_40_0.846.bin
│   ├── model_depth_15_estimator_100_0.855.bin
│   ├── model_depth_15_estimator_10_0.851.bin
│   ├── model_depth_15_estimator_20_0.853.bin
│   ├── model_depth_15_estimator_60_0.854.bin
│   ├── model_depth_20_estimator_20_0.857.bin
│   └── model_depth_25_estimator_60_0.858.bin
├── Dockerfile
├── README.md
├── client01.py
├── credit_risk_prediction.ipynb
├── model_training.py
├── predict.py
├── pyproject.toml
├── requirements.txt
└── uv.lock
```

Jupyter Notebook for EDA, data preprocessing, model training, hyper parameter tuning
- [credit_risk_prediction.ipynb](https://github.com/CarlosKim94/credit_risk_prediction/blob/main/credit_risk_prediction.ipynb)

Python script for data pre-processing and training
- [model_training.py](https://github.com/CarlosKim94/credit_risk_prediction/blob/main/model_training.py)
  
---

## Requirements & Dependencies
**Python Version:** 3.12 or above  

[Dependencies](https://github.com/CarlosKim94/credit_risk_prediction/blob/main/pyproject.toml):
- fastapi>=0.121.1
- matplotlib>=3.10.7
- numpy>=2.3.4
- pandas>=2.3.3
- requests>=2.32.5
- scikit-learn>=1.7.2
- seaborn>=0.13.2
- uvicorn>=0.38.0

Dependencies will all be automatically installed while deploying in the Docker container in the following section

---
## How to Run the Project

### 1. Clone the Repository

```bash
git clone git clone https://github.com/CarlosKim94/credit_risk_prediction.git
cd credit_risk_prediction
```

### 2. Create Virtual Environment

```bash
pip install uv
source .venv/bin/activate
```

To deactivate the virtual environment
```bash
deactivate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Jupyter Notebook

```bash
jupyter notebook
```

### 5. Reproduce Results

- Execute all cells in order in `credit_risk_prediction.ipynb` file
- Review plots, metrics, and feature importance
- Compare model performance in the results

### 6. Containerize and Deploy

- Install [Docker](https://www.docker.com/products/docker-desktop/) in your local machine
- Keep running Docker app in background
- Download the python image via:
```bash
docker pull python:3.12.12-bookworm
```

- Dockerfile uses `predict.py` which deploys the app via uvicorn and FastAPI
- Build and run the Dockerfile
```bash
docker build -t credit_risk_prediction .
docker run -it --rm -p 9696:9696 credit_risk_prediction
```

### 7. Test the Loan Default Prediction Model

- While Docker is still running, open a new terminal
- Change directory to `credit_risk_prediction`
- Activate virtual environment as in Step 2
```bash
source .venv/bin/activate
```

- Test the loan default prediction model with an arbitrary test data stored in `client01.py`
```bash
python client01.py
```
- Result would look like:
  
![prediction_test](https://github.com/CarlosKim94/credit_risk_prediction/blob/main/EDA/prediction_test.png)

- You can make a new test data similar to `client01.py` and compare how the prediction changes


---

## Acknowledgments

- Dataset: [Kaggle – Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)
- Libraries & Tools: Python, scikit-learn, Pandas, Seaborn, Numpy, fastAPI, uvicorn, Docker
- Inspiration: Financial risk analytics and applied data science research in credit scoring.
