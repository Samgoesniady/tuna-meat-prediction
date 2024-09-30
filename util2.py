import base64

import streamlit as st
from PIL import ImageOps, Image
import numpy as np
!pip install opencv-python
import cv2


def set_background(image_file):
    """
    This function sets the background of a Streamlit app to an image specified by the given image file.

    Parameters:
        image_file (str): The path to the image file to be used as the background.

    Returns:
        None
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)


def classify(image, model, class_names):
    """
    This function takes an image, a model, and a list of class names and returns the predicted class and confidence
    score of the image.

    Parameters:
        image (PIL.Image.Image): An image to be classified.
        model (tensorflow.keras.Model): A trained machine learning model for image classification.
        class_names (list): A list of class names corresponding to the classes that the model can predict.

    Returns:
        A tuple of the predicted class name and the confidence score for that prediction.
    """
    # convert image to (6000, 4000)
    image = ImageOps.fit(image, (6000, 4000), Image.Resampling.LANCZOS)

    # convert image to numpy array
    image_array = np.asarray(image)
    #image_array = image_array.transpose(1,0,2)

    # normalize image
    normalized_image_array = (image_array.astype(np.float32) / 255.0) - 1
    transposed_array = normalized_image_array.transpose(1,0,2)

    # set model input
    data = np.ndarray(shape=(1, 6000, 4000, 3), dtype=np.float32)
    data[0] = transposed_array #normalized_image_array

    # make prediction
    prediction = model.predict(data)
    
    # index = np.argmax(prediction)
    index = np.argmax(prediction[0])
    class_name = class_names[index]
    confidence_score = np.max(prediction[0])
    
    return class_name, confidence_score

def classify2(image, model2, class_names2):
    """
    This function takes an image, a model, and a list of class names and returns the predicted class and confidence
    score of the image.

    Parameters:
        image (PIL.Image.Image): An image to be classified.
        model (tensorflow.keras.Model): A trained machine learning model for image classification.
        class_names (list): A list of class names corresponding to the classes that the model can predict.

    Returns:
        A tuple of the predicted class name and the confidence score for that prediction.
    """
    # convert image to (6000, 4000)
    image = ImageOps.fit(image, (6000, 4000), Image.Resampling.LANCZOS)

    # convert image to numpy array
    image_array = np.asarray(image)
    #image_array = image_array.transpose(1,0,2)

    # normalize image
    normalized_image_array = (image_array.astype(np.float32) / 255.0) - 1
    transposed_array = normalized_image_array.transpose(1,0,2)

    # set model input
    data = np.ndarray(shape=(1, 6000, 4000, 3), dtype=np.float32)
    data[0] = transposed_array #normalized_image_array

    # make prediction
    prediction = model2.predict(data)
    
    # index = np.argmax(prediction)
    index = np.argmax(prediction[0])
    class_name2 = class_names2[index]
    confidence_score2 = np.max(prediction[0])
    
    return class_name2, confidence_score2
