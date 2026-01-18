import requests

url = 'http://localhost:8080/predict'

request_body = {
    "url": "https://raw.githubusercontent.com/CarlosKim94/pneumonia_detection/main/data/chest_xray_dataset/test/PNEUMONIA/person10_virus_35.jpeg"
}

response = requests.post(url, data=request_body)
result = response.json()

print(f"Top prediction: {result['prediction']} ({result['probability']:.2%})")