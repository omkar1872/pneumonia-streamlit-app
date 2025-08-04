import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import img_to_array

# Load the .keras model with EfficientNetB0 as a custom object
model = load_model("pneumonia_model.keras", custom_objects={"EfficientNetB0": EfficientNetB0})

# Streamlit UI
st.title("🩺 Pneumonia Detection from Chest X-ray")
st.markdown("Upload a chest X-ray image to detect signs of **pneumonia**.")

# Upload image
uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display image
    image_data = Image.open(uploaded_file).convert("RGB")
    st.image(image_data, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    image_resized = image_data.resize((224, 224))  # Adjust size if needed
    image_array = img_to_array(image_resized) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Make prediction
    prediction = model.predict(image_array)[0][0]

    # Display result
    if prediction > 0.5:
        st.error("Prediction: Pneumonia Detected 😷")
    else:
        st.success("Prediction: Normal Chest X-ray ✅")
