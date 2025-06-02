import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("model/pneumonia_cnn_model.h5")

def predict_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224)) / 255.0
    img = img.reshape(1, 224, 224, 1)
    prediction = model.predict(img)[0][0]
    return "Pneumonia Detected" if prediction > 0.5 else "Normal"

# Example usage:
# print(predict_image("images/test_xray.png"))
