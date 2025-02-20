import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("pneumonia_detection_model.h5")

# Class labels
class_names = ['NORMAL', 'PNEUMONIA']

# Function to preprocess and predict an image
def predict_image(image):
    img = image.resize((150, 150))  # Resize to match model input
    img = img.convert("RGB")  # Ensure 3 channels for CNN models
    
    img_array = img_to_array(img) / 255.0  # Normalize to [0,1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    prediction = model.predict(img_array)
    
    # Determine class
    predicted_class = class_names[int(prediction[0] > 0.5)]
    confidence = prediction[0][0] if predicted_class == 'PNEUMONIA' else 1 - prediction[0][0]

    return predicted_class, confidence

# Streamlit UI
st.set_page_config(page_title="Pneumonia Detector", layout="centered")

st.title("🩺 Pneumonia Detection System")
st.write("Upload a chest X-ray image to check for pneumonia.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Predict button
    if st.button("Analyze Image"):
        predicted_class, confidence = predict_image(image)
        st.write(f"### 🔍 Prediction: **{predicted_class}**")
        st.write(f"### 📊 Confidence: **{confidence:.2f}**")
