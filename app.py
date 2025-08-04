import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
from tensorflow.keras.applications import EfficientNetB0  # Add this line

# Load the model with custom_objects
model = load_model("pneumonia_efficientnetb0.h5", custom_objects={'EfficientNetB0': EfficientNetB0})

st.title("Pneumonia Detection from Chest X-ray")
uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = img.resize((224, 224))  # Size used during training
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    prediction = model.predict(x)
    if prediction[0][0] > 0.5:
        st.error("Prediction: Pneumonia Detected 😷")
    else:
        st.success("Prediction: Normal Chest X-ray ✅")
