import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Load trained CNN model
model = load_model("cnn_mnist_model.h5")

st.title("Handwritten Greyscale Digit Recognition🖊️")
st.write("Upload a digit image and the model will predict the number (0–9).")

# File uploader
uploaded_file = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])

def preprocess_image(img):
    # Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Resize to 28x28
    img_resized = cv2.resize(img_gray, (28, 28))
    # Invert if background is white
    if np.mean(img_resized) > 127:
        img_resized = 255 - img_resized
    # Normalize
    img_norm = img_resized / 255.0
    img_input = img_norm.reshape(1, 28, 28, 1)
    return img_resized, img_input

def predict_digit(img):
    img_resized, img_input = preprocess_image(img)
    # Predict
    prediction = model.predict(img_input)
    digit = np.argmax(prediction)
    confidence = np.max(prediction)
    return digit, confidence, img_resized

# Main logic
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="🖼️ Uploaded Image", use_container_width=True)

    digit, confidence, resized_img = predict_digit(img)

    # Show the 28x28 resized image (scaled up for visibility)
    resized_display = cv2.resize(resized_img, (280, 280), interpolation=cv2.INTER_NEAREST)
    st.image(resized_display, caption="🔍 Resized 28×28 Image (used for prediction)", use_container_width=False)

    # Confidence threshold
    if confidence < 0.7:
        st.error("⚠️ Invalid image or not a handwritten digit!")
    else:
        st.success(f"✅ Predicted Digit: **{digit}**")
