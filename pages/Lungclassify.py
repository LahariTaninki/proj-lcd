import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input

# Set page configuration
st.set_page_config(
    page_title="Vitality Lung Predictor",
    initial_sidebar_state="expanded",
)

# Hide Streamlit footer and menu
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Choose a File", type=['jpg', 'png', 'jpeg'])

# Load the model
model = load_model('models/LCC.h5', compile=False)

# Function to preprocess the image
def preprocess_image(image):
    size = (224, 224)
    image = image.convert("RGB")  # Convert RGBA image to RGB
    image = ImageOps.fit(image, size)
    image_array = np.asarray(image)
    image_array = np.expand_dims(image_array, axis=0)
    return preprocess_input(image_array)

# Custom class labels
custom_class_labels = ["Adenocarcinoma", "Large cell Carcinoma", "Normal", "Squamous"]

# Predict button
if uploaded_file is not None:
    st.image(uploaded_file)
    predict_button = st.button("Predict")
    if predict_button:
        with st.spinner("Diagnosing..."):
            image = Image.open(uploaded_file)
            processed_image = preprocess_image(image)
            predictions = model.predict(processed_image)
            predicted_class_index = np.argmax(predictions)
            if 0 <= predicted_class_index < len(custom_class_labels):
                predicted_class_label = custom_class_labels[predicted_class_index]
                st.write("Predicted Class Index:", predicted_class_index)
                st.write("Predicted Class Label:", predicted_class_label)
            else:
                st.error("Error: Predicted class index is out of range.")
