import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier  # Import Random Forest classifier
import matplotlib.pyplot as plt
import cv2
import os
import joblib

dataset_dir = '/content/dataset'

# Define global variables for labels
benign_labels = None
malignant_labels = None

# Step 1: Data Collection (Replace with your dataset)
# Example directories for benign and malignant images
benign_dir = '/content/dataset/benign'
malignant_dir = '/content/dataset/malignant'

# Step 2: Preprocessing
# Load and preprocess the images
def load_images_from_directory(directory):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            img = cv2.imread(os.path.join(directory, benign_dir))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)  # Convert to grayscale and set data type to 32-bit float
            img = cv2.resize(img, (224, 224))  # Resize to a standard size
            img = img / 255.0  # Normalize pixel values
            images.append(img)
    return images

benign_images = load_images_from_directory(benign_dir)
malignant_images = load_images_from_directory(malignant_dir)

# Step 3: Feature Extraction
# Load a pretrained VGG model for feature extraction
base_model = VGG16(weights='imagenet', include_top=False)

# Define a function to preprocess and prepare the images for feature extraction
def preprocess_images(images):
    preprocessed_images = []
    for image in images:
        # Expand dimensions to (height, width, 3) for RGB channels
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # Resize and preprocess
        image_rgb = cv2.resize(image_rgb, (224, 224))
        image_rgb = image_rgb / 255.0  # Normalize pixel values
        preprocessed_images.append(image_rgb)
    return np.array(preprocessed_images)

# Extract features
benign_features = base_model.predict(preprocess_images(benign_images))
malignant_features = base_model.predict(preprocess_images(malignant_images))

# Set labels in the global scope
benign_labels = [0] * len(benign_features)
malignant_labels = [1] * len(malignant_features)

# Combine features and labels
X = np.concatenate((benign_features, malignant_features), axis=0)
y = np.array(benign_labels + malignant_labels)

# Step 4: Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Flatten the feature vectors
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Step 5: Classification Model
# Train a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_flat, y_train)

# Step 6: Evaluation
y_pred_rf = rf_classifier.predict(X_test_flat)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
report_rf = classification_report(y_test, y_pred_rf)

print(f'Random Forest Accuracy: {accuracy_rf}')
print(report_rf)

# Step 7: Prediction and Mapping
y_pred_rf = rf_classifier.predict(X_test_flat)

# Map the predicted values to 0 for benign and 1 for malignant
y_pred_mapped = [0 if pred == 0 else 1 for pred in y_pred_rf]

print("Mapped Predictions:")
print(y_pred_mapped)

# Save the RandomForestClassifier model using joblib
joblib.dump(rf_classifier, 'BreastCancerRF.pkl')