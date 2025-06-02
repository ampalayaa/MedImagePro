from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Load and preprocess data
datagen = ImageDataGenerator(rescale=1./255)

train = datagen.flow_from_directory("chest_xray/train", target_size=(224, 224), color_mode='grayscale', class_mode='binary')
val = datagen.flow_from_directory("chest_xray/val", target_size=(224, 224), color_mode='grayscale', class_mode='binary')

# Build CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train, validation_data=val, epochs=10)

# Save model
os.makedirs("model", exist_ok=True)
model.save("model/pneumonia_cnn_model.h5")
