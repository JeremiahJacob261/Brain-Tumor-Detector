import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import streamlit as st


#streamlit components
st.title('My First Streamlit App')

st.write('Welcome to my app! Below is a simple input form.')
# Define paths
data_dir_yes = 'brain_tumor_dataset/yes'
image_path = os.path.join(data_dir_yes, 'Y3.jpg')

# Load the trained model (assuming the model was saved as 'brain_tumor_model.h5')
model = load_model('brain_tumor_model.h5')


# Load and preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is not None:
        img = cv2.resize(img, (64, 64))  # Resize to match model input size
        img = img / 255.0  # Normalize
        img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img
uploaded_files = st.file_uploader("Choose an Image file", accept_multiple_files=False)
for uploaded_file in uploaded_files:
    st.write("filename:", uploaded_file.name)
    st.image(uploaded_file)
    current_directory = os.getcwd()
    image_path = os.path.join(current_directory, uploaded_file.name)
    img = preprocess_image(image_path)

    # Make a prediction
    prediction = model.predict(img)

    # Get the probability of the image being a tumor
    tumor_probability = prediction[0][1]  # Probability of the class 'yes' (index 1)
    st.write(f'Probability of the image being a tumor: {tumor_probability * 100:.2f}%')
    # print(f'Probability of the image being a tumor: {tumor_probability * 100:.2f}%')
