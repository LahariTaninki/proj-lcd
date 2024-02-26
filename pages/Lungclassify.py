import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input

# Define the Streamlit page configuration
st.set_page_config(
    page_title="VLP",
    initial_sidebar_state="expanded",
)

# Hide the Streamlit menu and footer
st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)

# Load the lung cancer prediction model
model = load_model('models/LCC.h5', compile=False)

# Define custom class labels for prediction
custom_class_labels = ["Adenocarcinoma", "Large cell Carcinoma", "Normal", "Squamous"]

# Title of the application
st.title("Vitality Lung Predictor")

# File uploader for user to upload an image
uploaded_file = st.file_uploader("Choose a File", type=['jpg', 'png', 'jpeg'])

# Button to trigger the prediction process
if st.button("Predict"):
    # Check if an image is uploaded
    if uploaded_file is not None:
        # Open and preprocess the uploaded image
        image = Image.open(uploaded_file)
        size = (224, 224)
        image = ImageOps.fit(image, size)
        image_array = np.asarray(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = preprocess_input(image_array)
        
        # Make predictions using the loaded model
        predictions = model.predict(image_array)
        predicted_class_index = np.argmax(predictions)
        predicted_class_label = custom_class_labels[predicted_class_index]

        # Display the predicted class index and label
        st.write("Predicted Class Index:", predicted_class_index)
        st.write("Predicted Class Label:", predicted_class_label)
    else:
        # Display a message if no image is uploaded
        st.error("Please upload an image first.")
