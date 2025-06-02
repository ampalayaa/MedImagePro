import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

model = load_model(os.path.join("..", "model", "pneumonia_cnn_model.h5"))

def predict_image_from_array(img_array):
    img = cv2.resize(img_array, (224, 224)) / 255.0
    img = img.reshape(1, 224, 224, 1)
    pred = model.predict(img)[0][0]
    return "Pneumonia Detected" if pred > 0.5 else "Normal"

class PneumoniaApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Pneumonia Detector")
        self.geometry("600x400")

        self.select_button = ctk.CTkButton(self, text="Select X-ray Image", command=self.load_image)
        self.select_button.pack(pady=20)

        self.image_label = ctk.CTkLabel(self, text="No Image Loaded")
        self.image_label.pack(pady=10)

        self.result_label = ctk.CTkLabel(self, text="")
        self.result_label.pack(pady=10)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        if file_path:
            img_array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            result = predict_image_from_array(img_array)
            img = Image.fromarray(img_array)
            img = img.resize((200, 200))
            img_tk = ImageTk.PhotoImage(img)
            self.image_label.configure(image=img_tk, text="")
            self.image_label.image = img_tk
            self.result_label.configure(text=result)

if __name__ == "__main__":
    app = PneumoniaApp()
    app.mainloop()
