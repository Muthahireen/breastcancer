
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img

# Load the model from the HDF5 file
model = load_model('/content/BreastCancer1.h5')

# Function to preprocess the image (resize to the model's expected input shape)
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Load and preprocess the new image (replace 'path/to/new_image.jpg' with your image path)
new_image_path = '.png'
new_image = preprocess_image(new_image_path)

# Make predictions
predictions = model.predict(new_image)

# Get the predicted class (assuming binary classification)
predicted_class = 0 if predictions[0][0] > 0.5 else 1

# Display the predicted class
print("Predicted class:", predicted_class)