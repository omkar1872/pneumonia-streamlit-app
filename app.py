import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the model saved in modern format (.keras)
model = load_model("pneumonia_model.keras")

st.title("Pneumonia Detection from Chest X-ray")
uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = img.resize((224, 224))  # Resize to match model input
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0  # Normalize

    # Predict
    prediction = model.predict(x)

    # Output result
    if prediction[0][0] > 0.5:
        st.error("Prediction: Pneumonia Detected 😷")
    else:
        st.success("Prediction: Normal Chest X-ray ✅")
