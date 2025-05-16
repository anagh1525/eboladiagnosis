import os
import xml.etree.ElementTree as ET
import tensorflow as tf
import tensorflowjs as tfjs
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import json

# === CONFIG ===
IMAGE_DIR = "images"
ANNOTATION_DIR = "annotations"
EXPORT_DIR = "tfjs_model"
IMG_SIZE = (224, 224)
LABEL_MAP = {"ebola": 0, "not_ebola": 1}

# Load Image + Label from XML 
def load_data_from_pascal_voc(image_dir, annotation_dir):
    X = []
    y = []
    for filename in os.listdir(annotation_dir):
        if not filename.endswith(".xml"):
            continue
        xml_path = os.path.join(annotation_dir, filename)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        image_name = root.find("filename").text
        image_path = os.path.join(image_dir, image_name)
        if not os.path.exists(image_path):
            continue

        obj = root.find("object")
        label = obj.find("name").text.strip().lower()

        if label not in LABEL_MAP:
            continue

        # Load and resize image
        img = Image.open(image_path).convert("RGB")
        img = img.resize(IMG_SIZE)
        X.append(np.array(img) / 255.0)
        y.append(LABEL_MAP[label])

    return np.array(X), np.array(y)

# Load Data 
X, y = load_data_from_pascal_voc(IMAGE_DIR, ANNOTATION_DIR)
y = tf.keras.utils.to_categorical(y, num_classes=2)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define CNN 
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train 
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=16)

# Export to TensorFlow.js format 
if not os.path.exists(EXPORT_DIR):
    os.makedirs(EXPORT_DIR)
tfjs.converters.save_keras_model(model, EXPORT_DIR)

#  Create metadata.json
metadata = {
    "labels": ["ebola", "not_ebola"],
    "imageSize": IMG_SIZE[0],
    "modelType": "image",
    "modelName": "Ebola Classifier"
}
with open(os.path.join(EXPORT_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=4)

print("✅ Model and metadata exported to tfjs_model/")
