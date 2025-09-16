import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load your trained model
MODEL_PATH = "models/fine_tuned_model.h5"  # relative path inside repo
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels
class_labels = ["covid", "pneumonia", "tb", "normal"]

st.set_page_config(page_title="Chest X-Ray Classifier", page_icon="🩺", layout="centered")
st.title("🩺 Chest X-Ray Disease Classifier")
st.write("Upload a chest X-ray image and the model will predict if it is **COVID, Pneumonia, TB, or Normal**.")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded X-ray", use_column_width=True)

    # Preprocess
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    pred_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.markdown(f"### ✅ Prediction: **{pred_class.upper()}** ({confidence:.2f}%)")

    st.subheader("Class Probabilities:")
    for label, prob in zip(class_labels, prediction[0]):
        st.write(f"- {label}: {prob*100:.2f}%")
