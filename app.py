import streamlit as st
import tensorflow as tf
from tf.keras.models import load_model
import numpy as np
from PIL import Image
import io



# Load the saved model
model = load_model(r'C:\Users\HP\Desktop\DATALAB\pythonclasses\medical_image_classifier_new\saved_model\my_model.h5')

# Define the classes based on your dataset
class_names = ['Healthy', 'Malignant']  # Update this based on your dataset


# Function to preprocess the uploaded image
#def preprocess_image(image):
    
    
    #image_array = np.array(image)
    #image_array = image_array / 255.0  # Normalize the image
    #image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    #return image_array


def preprocess_image(image):
    # Convert the image to RGB (if it's a PNG with transparency or grayscale)
    image = image.convert('RGB')
    # Resize the image to match the input shape of the model
    image = image.resize((224, 224))  # Adjust based on your model's input shape
    # Convert to numpy array and normalize
    image_array = np.array(image) / 255.0  # Normalize to [0, 1]
    # Add a batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Streamlit User Interface
st.title("Medical Image Classification")
st.write("This app uses a pre-trained model to classify MRI images into various categories.")
st.write("You can upload your own MRI image and see the predicted label.")
st.write("Upload a medical image for classification.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image')

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make prediction
    predictions = model.predict(processed_image)
    predicted_class = 'Malignant' if predictions[0][0] > 0.5 else 'Healthy'


    # Display the result
    st.write(f"Predictions: {predicted_class}")


imgs = Image.open("mri3.gif")
st.sidebar.image(imgs)

st.sidebar.button("Reset", type="secondary")
if st.sidebar.button('Predicted_Class'):
    predictions
    