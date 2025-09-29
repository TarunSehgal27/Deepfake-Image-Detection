import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
model = load_model("models/hybrid_model.h5")

def preprocess_img(img):
    img = cv2.resize(image, (224,224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img/255.0
    return img

def predict_img(img_path):
    img = predict_img(img_path)
    prediction = model.predict(img)
    class_label = np.argmax(prediction, axis=1)[0]
    return "Real" if class_label==1 else "Fake"
