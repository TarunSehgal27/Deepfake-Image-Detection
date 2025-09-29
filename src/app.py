import streamlit as st
import numpy as np 
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

def preprocess_img(img):
    img = cv2.resize(image, (224,224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img/255.0
    return img

img = Image.open("img/page_icon.jfif")
st.set_page_config(page_title="Fake Image Detector", page_icon=img, initial_sidebar_state="expanded")
st.header("Deepfake Image Detection Tool")

with st.sidebar:
  st.image("img/detector.png")
  st.write("---")
  st.title("About")
  st.write(
    """This application represents the culmination of advanced research in computer vision and machine learning, specifically
    designed to stay ahead of evolving deepfake generation techniques. Our model continuously learns from new patterns and
    maintains effectiveness against emerging manipulation methods. The deepfake media circulating on the social media applications 
    can be checked using this tool.
  """)

  st.markdown(
            """
            ---
            Follow me on:

            Github → [@tarunsehgal27](https://github.com/tarunsehgal27)

            LinkedIn → [Tarun Sehgal](https://www.linkedin.com/in/tarunsehgal27)

            """
        )

uploaded_file = st.file_uploader("Choose your file...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    st.write("File Uploaded Successfully")
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, channels="BGR")

st.subheader("""Challenge We Address""")
st.write(
    """Deepfake and AI generated images have become increasingly sophisticated, making it difficult to distinguish authentic
    content from manipulated media with naked eye. Traditional detection methods fall shot when faced with advanced generation
    techniques like GANs, Diffusion models etc. creating a need for more powerful detection solutions.
    """)

st.subheader("""Hybrid GAN-ResNet152 Architecture""")
st.write(
    """This application is built on groundbreaking research that combines two powerful model architectures:
    GANs used for sophisticated deepfake creation.
    ResNet for feature extraction and analysis.
    """)

st.subheader("Model Training Graph")
st.image("img/model_accuracy.png")
st.markdown("Model Accuracy: 88%")

st.markdown(
        """
        <div style="position: float; bottom: 0; left: 0; width: 100%; padding: 15px; text-align: center;">
            © <a href="https://github.com/tarunsehgal27" target="_blank">Tarun Sehgal</a> | <strong>Made with ❤️ </strong>
        </div>
        """,
        unsafe_allow_html=True
    )