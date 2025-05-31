import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk
from tkinter import filedialog
import tensorflow as tf

class MiniMedImagePro(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Mini MedImagePro")
        self.geometry("1000x600")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        # Custom yellow theme color
        self.button_color = "#f1c40f"

        # Left sidebar frame for buttons
        self.button_frame = ctk.CTkFrame(self, width=180, fg_color="#2d2d2d")
        self.button_frame.pack(side="left", fill="y", padx=5, pady=5)

        # Image display panel
        self.image_panel = ctk.CTkLabel(self, text="No Image", anchor="center")
        self.image_panel.pack(expand=True, fill="both", padx=10, pady=10)

        # Diagnostic result label
        self.diagnosis_label = ctk.CTkLabel(self, text="Diagnosis: None", anchor="center", text_color="white")
        self.diagnosis_label.pack(fill="x", padx=10, pady=(0, 10))

        # Buttons
        ctk.CTkButton(self.button_frame, text="Open Image", command=self.open_image, fg_color=self.button_color, text_color="black").pack(fill="x", pady=5, padx=10)
        ctk.CTkButton(self.button_frame, text="Predict Diagnosis", command=self.predict_diagnosis, fg_color=self.button_color, text_color="black").pack(fill="x", pady=5, padx=10)
        ctk.CTkButton(self.button_frame, text="Reset", command=self.reset_image, fg_color=self.button_color, text_color="black").pack(fill="x", pady=5, padx=10)

        self.original_image = None
        self.processed_image = None
        self.model = tf.keras.models.load_model('pneumonia_detector.h5')
        self.img_width, self.img_height = 150, 150

    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if path:
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.original_image = image
            self.processed_image = image.copy()
            self.display_image(image)
            self.diagnosis_label.configure(text="Diagnosis: None")

    def reset_image(self):
        if self.original_image is not None:
            self.processed_image = self.original_image.copy()
            self.display_image(self.original_image)
            self.diagnosis_label.configure(text="Diagnosis: None")

    def display_image(self, img_array):
        img = Image.fromarray(img_array)
        img = img.resize((512, 512))
        photo = ImageTk.PhotoImage(img)
        self.image_panel.configure(image=photo, text="")
        self.image_panel.image = photo

    def predict_diagnosis(self):
        if self.processed_image is not None:
            img = cv2.resize(self.processed_image, (self.img_width, self.img_height))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)
            prediction = self.model.predict(img)[0][0]
            if prediction > 0.5:
                diagnosis = "PNEUMONIA"
                confidence = prediction
            else:
                diagnosis = "NORMAL"
                confidence = 1 - prediction
            self.diagnosis_label.configure(text=f"Diagnosis: {diagnosis} ({confidence*100:.2f}% confidence)")
