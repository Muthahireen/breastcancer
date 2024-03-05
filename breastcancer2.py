import numpy as np
import cv2
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.externals import joblib

def load_images_from_directory(directory):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            img = cv2.imread(os.path.join(directory, filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)  
            img = cv2.resize(img, (224, 224))  
            img = img / 255.0 
            images.append(img)
    return images

def extract_features(base_model, images):
    features = []
    for image in images:
        image_rgb = cv2.resize(image, (224, 224))
        image_rgb = image_rgb / 255.0
        features.append(base_model.predict(np.array([image_rgb])))
    return np.vstack(features)

def preprocess_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    return X_train_flat, X_test_flat, y_train, y_test

def train_svm_classifier(X_train_flat, y_train):
    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(X_train_flat, y_train)
    return svm_classifier

def evaluate_svm_classifier(svm_classifier, X_test_flat, y_test):
    y_pred = svm_classifier.predict(X_test_flat)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

def save_svm_classifier(svm_classifier, model_filename):
    joblib.dump(svm_classifier, model_filename)

# Step 1: Data Collection
benign_dir = '/content/dataset/benign'
malignant_dir = '/content/dataset/malignant'

# Step 2: Preprocessing
benign_images = load_images_from_directory(benign_dir)
malignant_images = load_images_from_directory(malignant_dir)

# Step 3: Feature Extraction

features = base_model.predict(np.array([image]))
benign_features = extract_features(base_model, benign_images)
malignant_features = extract_features(base_model, malignant_images)

# Create labels
benign_labels = [0] * len(benign_features)
malignant_labels = [1] * len(malignant_features)

# Combine features and labels
X = np.concatenate((benign_features, malignant_features), axis=0)
y = np.array(benign_labels + malignant_labels)

# Step 4: Data Splitting
X_train_flat, X_test_flat, y_train, y_test = preprocess_data(X, y)

# Step 5: Classification Model
svm_classifier = train_svm_classifier(X_train_flat, y_train)

# Step 6: Evaluation
accuracy, report = evaluate_svm_classifier(svm_classifier, X_test_flat, y_test)

print(f'Accuracy: {accuracy}')
print(report)

# Step 7: Export the Model as H5 file
model_filename = 'BreastCancerSVM.joblib'
save_svm_classifier(svm_classifier, model_filename)