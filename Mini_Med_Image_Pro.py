import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from io import BytesIO

class MiniMedImagePro(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Mini MedImagePro - Educational Toolkit")
        self.geometry("1200x700")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        # Custom yellow theme color
        self.button_color = "#f1c40f"

        # Left sidebar frame for buttons
        self.button_frame = ctk.CTkFrame(self, width=200, fg_color="#2d2d2d")
        self.button_frame.pack(side="left", fill="y", padx=5, pady=5)

        # Image display frames for original and processed images
        self.original_frame = ctk.CTkFrame(self, fg_color="#2d2d2d")
        self.original_frame.pack(side="left", expand=True, fill="both", padx=5, pady=5)
        
        self.processed_frame = ctk.CTkFrame(self, fg_color="#2d2d2d")
        self.processed_frame.pack(side="left", expand=True, fill="both", padx=5, pady=5)

        # === Added labels above each image panel ===
        self.original_title = ctk.CTkLabel(self.original_frame, text="Original X-ray Image", font=("Arial", 16, "bold"))
        self.original_title.pack(pady=(10, 0))

        self.original_label = ctk.CTkLabel(self.original_frame, text="Original Image", anchor="center")
        self.original_label.pack(expand=True, fill="both", padx=10, pady=10)

        self.processed_title = ctk.CTkLabel(self.processed_frame, text="Processed X-ray Image", font=("Arial", 16, "bold"))
        self.processed_title.pack(pady=(10, 0))

        self.processed_label = ctk.CTkLabel(self.processed_frame, text="Processed Image", anchor="center")
        self.processed_label.pack(expand=True, fill="both", padx=10, pady=10)

        # Add buttons with tooltips and improved layout
        buttons = [
            ("Open Image", self.open_image, "Load a grayscale chest X-ray image (.png, .jpg, .jpeg)"),
            ("Histogram Equalization", self.histogram_equalization, "Enhance contrast to reveal hidden details"),
            ("Median Filter", self.median_filter, "Reduce salt-and-pepper noise while preserving edges"),
            ("Mean Filter", self.mean_filter, "Smooth the image by averaging pixel values"),
            ("Sobel Edge Detection", self.sobel_edge, "Highlight anatomical boundaries like ribs and lungs"),
            ("Show Histogram", self.show_histogram, "Display pixel intensity distribution"),
            ("Save Processed Image", self.save_image, "Save the processed image to your device"),
            ("Reset", self.reset_image, "Restore the original image")
        ]
        
        for text, command, tooltip in buttons:
            btn = ctk.CTkButton(self.button_frame, text=text, command=command, 
                               fg_color=self.button_color, text_color="black")
            btn.pack(fill="x", pady=8, padx=10)
            ctk.CTkLabel(self.button_frame, text=tooltip, font=("Arial", 10), 
                        wraplength=170, text_color="#cccccc").pack(pady=2)

        self.original_image = None
        self.processed_image = None
        self.display_size = (450, 450)  # Standard display size for consistency

    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if path:
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                messagebox.showerror("Error", "Failed to load image. Ensure it's a valid file.")
                return
            self.original_image = image
            self.processed_image = image.copy()
            self.display_image(self.original_image, self.original_label)
            self.display_image(self.processed_image, self.processed_label)
            messagebox.showinfo("Success", "Grayscale chest X-ray image loaded successfully!")

    def reset_image(self):
        if self.original_image is not None:
            self.processed_image = self.original_image.copy()
            self.display_image(self.processed_image, self.processed_label)
            messagebox.showinfo("Reset", "Image restored to original state.")

    def display_image(self, img_array, label):
        if img_array is None:
            return
        img = Image.fromarray(img_array)
        img = img.resize(self.display_size, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        label.configure(image=photo, text="")
        label.image = photo

    def histogram_equalization(self):
        if self.processed_image is not None:
            self.processed_image = cv2.equalizeHist(self.processed_image)
            self.display_image(self.processed_image, self.processed_label)
            messagebox.showinfo("Applied", "Histogram Equalization: Contrast enhanced!")

    def median_filter(self):
        if self.processed_image is not None:
            self.processed_image = cv2.medianBlur(self.processed_image, 5)
            self.display_image(self.processed_image, self.processed_label)
            messagebox.showinfo("Applied", "Median Filter: Noise reduced!")

    def mean_filter(self):
        if self.processed_image is not None:
            kernel = np.ones((5, 5), np.float32) / 25
            self.processed_image = cv2.filter2D(self.processed_image, -1, kernel)
            self.display_image(self.processed_image, self.processed_label)
            messagebox.showinfo("Applied", "Mean Filter: Image smoothed!")

    def sobel_edge(self):
        if self.processed_image is not None:
            sobelx = cv2.Sobel(self.processed_image, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(self.processed_image, cv2.CV_64F, 0, 1, ksize=3)
            sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
            self.processed_image = np.clip(sobel, 0, 255).astype(np.uint8)
            self.display_image(self.processed_image, self.processed_label)
            messagebox.showinfo("Applied", "Sobel Edge Detection: Edges highlighted!")

    def show_histogram(self):
        if self.processed_image is not None:
            plt.figure(figsize=(6, 4))
            plt.hist(self.processed_image.ravel(), bins=256, range=(0, 255), color='gray')
            plt.title("Pixel Intensity Histogram")
            plt.xlabel("Pixel Value")
            plt.ylabel("Frequency")
            plt.grid(True)
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight')
            plt.close()
            buffer.seek(0)
            img = Image.open(buffer)
            img = img.resize(self.display_size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.processed_label.configure(image=photo, text="")
            self.processed_label.image = photo
            messagebox.showinfo("Histogram", "Displaying pixel intensity distribution!")

    def save_image(self):
        if self.processed_image is not None:
            path = filedialog.asksaveasfilename(defaultextension=".png", 
                                               filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")])
            if path:
                cv2.imwrite(path, self.processed_image)
                messagebox.showinfo("Saved", "Processed image saved successfully!")

if __name__ == "__main__":
    app = MiniMedImagePro()
    app.mainloop()
