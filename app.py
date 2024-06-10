import numpy as np
import streamlit as st
import cv2
from keras.models import load_model

# Load the model
model = load_model('plant_disease.h5')

# Name of Classes
CLASS_NAMES = ['Corn-Common_rust', 'Potato-Early_blight', 'Tomato-Bacterial_spot']

# Inject CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("styles.css")

# Setting Title of App
st.markdown('<h1 class="main-title">Plant Disease Detection</h1>', unsafe_allow_html=True)
st.markdown('<p class="middle-title">Upload an image of the plant leaf</p>', unsafe_allow_html=True)

# Uploading the plant image
plant_image = st.file_uploader("Choose an image...", type="jpg")
submit = st.button('Predict', key='predict_button', help="Click to predict the disease")

# On predict button click
if submit:
    if plant_image is not None:
        # Convert the file to an opencv image
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        # Displaying the image
        st.image(opencv_image, channels="BGR")
        st.write(opencv_image.shape)
        
        # Resizing the image
        opencv_image = cv2.resize(opencv_image, (256,256))
        
        # Convert image to 4 Dimensions
        opencv_image.shape = (1,256,256,3)
        
        # Make Prediction
        Y_pred = model.predict(opencv_image)
        result = CLASS_NAMES[np.argmax(Y_pred)]
        
        st.markdown(f'<h3 class="result-title">This is a {result.split("-")[0]} leaf with {result.split("-")[1]}</h3>', unsafe_allow_html=True)
