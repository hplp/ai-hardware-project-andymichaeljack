# testModel.py
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# Paths
model_path = 'traffic_classifier_model.h5'
test_dir = 'data/test'

# Load the Model
model = load_model(model_path)

# Function to Predict Traffic Level
def predict_image(image_path):
    img = load_img(image_path, target_size=(128, 128))  # Load and resize image
    img_array = img_to_array(img) / 255.0              # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)      # Add batch dimension
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)                 # Get the index of the highest prediction
    class_labels = {0: "Low", 1: "Medium", 2: "High"}  # Map index to class labels
    return class_labels[class_idx]

# Test on Directory
def test_directory(test_dir):
    for category in os.listdir(test_dir):  # Iterate through class folders
        category_path = os.path.join(test_dir, category)
        if os.path.isdir(category_path):
            print(f"Category: {category}")
            for image_file in os.listdir(category_path):
                image_path = os.path.join(category_path, image_file)
                result = predict_image(image_path)
                print(f"{image_file}: Predicted as {result}")

# Test Images
test_directory(test_dir)
