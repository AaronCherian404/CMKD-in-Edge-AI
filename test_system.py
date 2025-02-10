import requests
import torch
from PIL import Image
import json

def test_inference():
    # Test edge service
    image_path = "test_images/sample.jpg"
    image = Image.open(image_path)
    
    # Convert image to tensor and prepare payload
    transform = transforms.ToTensor()(image)
    payload = {"data": transform.tolist()}
    
    # Test edge inference
    response = requests.post(
        "http://localhost:8002/inference",
        json=payload
    )
    
    print("Edge Service Response:", response.json())

if __name__ == "__main__":
    test_inference()