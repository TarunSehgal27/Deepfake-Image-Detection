# Deepfake-Image-Detection
A real-time deepfake image detection system built with a custom deep learning model trained from scratch. Users can upload images through an intuitive Streamlit interface to classify them as real or AI-generated/manipulated. This application represents the culmination of advanced research in computer vision and machine learning, specifically designed to stay ahead of evolving deepfake generation techniques. The deepfake media circulating on the social media applications can be checked using this tool. This application is built on groundbreaking research that combines two powerful model architectures:
- **GANs**: GANs used for sophisticated deepfake creation.
- **ResNet**: ResNet for feature extraction and analysis.

## Code
Check out the code of the project by clicking [here](https://github.com/TarunSehgal27/Deepfake-Image-Detection/tree/main/src)!

## Features
- Custom-built deep learning model for binary classification (Real/Fake)
- User-friendly web interface powered by Streamlit
- Real-time image analysis and prediction
- Model stored in H5 format for efficient deployment
  

## How to run
Clone the repository to your local machine:
```
git clone https://github.com/TarunSehgal27/Deepfake-Image-Detection.git
```
Install the required dependencies:
```
pip install -r requirements.txt
```
Run the Streamlit app:
```
streamlit run app.py
```
## Tech Stack
- **Frontend:** Streamlit
- **Backend:** Python
- **Model Format:** HDF5 (.h5)
- **Deep Learning Framework:** TensorFlow/Keras

