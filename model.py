import tensorflow as tf
from tensorflow.keras.models import load_model  # Import load_model from Keras

# Define a function to load the saved model
def saved_model():
    model = load_model (r'C:\Users\HP\Desktop\DATALAB\pythonclasses\medical_image_classifier\saved_model\my_model.h5')  # Update with your actual path
    return model

