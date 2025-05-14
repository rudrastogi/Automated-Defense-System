import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

# Dataset path
DATASET_PATH = r"C:\Users\aryan\OneDrive\Desktop\Face Recognition\dataset"
IMG_SIZE = (100, 100)  # Image size for resizing
CATEGORIES = [category for category in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, category))]


# Load dataset
def load_dataset():
    images, labels = [], []
    label_map = {category: idx for idx, category in enumerate(CATEGORIES)}

    for category in CATEGORIES:
        path = os.path.join(DATASET_PATH, category)
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            if os.path.isfile(img_path):  # Ensure it's a file
                image = cv2.imread(img_path)
                if image is not None:
                    image = cv2.resize(image, IMG_SIZE)  # Resize image
                    image = image / 255.0  # Normalize pixel values
                    images.append(image)
                    labels.append(label_map[category])

    return np.array(images), np.array(labels), label_map


# Load dataset
images, labels, label_map = load_dataset()
labels = to_categorical(labels, num_classes=len(CATEGORIES))

# Split data
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)


# Define CNN model
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(len(CATEGORIES), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Train model
model = create_model()

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model_checkpoint = ModelCheckpoint("best_face_recognition_model.h5", save_best_only=True)

history = model.fit(
    X_train, y_train, epochs=20, validation_data=(X_test, y_test),
    callbacks=[early_stopping, model_checkpoint]
)

# Save final model
MODEL_PATH = "face_recognition_model.h5"
model.save(MODEL_PATH)
print(f"Model saved at {MODEL_PATH}")

