import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, decode_predictions, preprocess_input
import numpy as np
from PIL import Image

# Load the ResNet50 pre-trained model
@st.cache_resource  # Cache the model for faster loading
def load_model():
    return ResNet50(weights='imagenet')

model = load_model()

# Streamlit app title
st.title("Image Classification Using ResNet50")
st.write("Upload an image, and the model will predict the top 3 most likely categories.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)  # Updated parameter

    # Preprocess the image
    img = Image.open(uploaded_file)
    img = img.resize((224, 224))  # ResNet50 expects 224x224 images
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess the image

    # Make a prediction
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    # Display the top 3 predictions
    st.subheader("Top 3 Predictions:")
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        st.write(f"{i + 1}. {label}: {score:.2f}")