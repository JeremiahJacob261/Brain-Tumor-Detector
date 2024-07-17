import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import streamlit as st


st.title('Brain Cancer Detector')
st.markdown(f'<p style="font-weight:bold;">An Undergraduate project by OMOTUYI SAMUEL OLAREWAJU</p>', unsafe_allow_html=True)
st.write('Upload an MRI Image Scan')
model = load_model('brain_tumor_model.h5')


# Load and preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is not None:
        img = cv2.resize(img, (64, 64))  # Resize to match model input size
        img = img / 255.0  # Normalize
        img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img
uploaded_files = st.file_uploader("Choose an Image file", accept_multiple_files=True)
for uploaded_file in uploaded_files:
    st.write("filename:", uploaded_file.name)
    st.image(uploaded_file)
    current_directory = os.getcwd()
    image_folder = os.path.join(current_directory,'testing_dataset')
    image_path = os.path.join(image_folder, uploaded_file.name)
    # st.write(image_path)
    img = preprocess_image(image_path)

    # Make a prediction
    prediction = model.predict(img)

    # Get the probability of the image being a tumor
    tumor_probability = prediction[0][1]  # Probability of the class 'yes' (index 1)
    tumor_percent = tumor_probability * 100
    if tumor_percent > 50:
        st.markdown(f'<div style="display:flex;flex-direction:column;padding:16px;background-color:#283845;border-radius:8px;border:0.6px solid #DDE1E4;"><div style="display:flex;flex-direction:row;"><p style="padding:6px;">Probability of the image being a tumor: </p><p style="font-size:24px;font-weight:600;color:red;">{tumor_probability * 100:.2f}%</p></div><p style="font-size:20px;font-weight:600;padding:5px;">Result: The Patient is likely to have Brain Cancer</p></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="display:flex;flex-direction:column;padding:16px;background-color:#283845;border-radius:8px;border:0.6px solid #DDE1E4;"><div style="display:flex;flex-direction:row;"><p style="padding:6px;">Probability of the image being a tumor: </p><p style="font-size:24px;font-weight:600;color:#5AFF15;">{tumor_probability * 100:.2f}%</p></div><p style="font-size:20px;font-weight:600;padding:5px;">Result: The Patient does not have Brain Cancer</p></div>', unsafe_allow_html=True)