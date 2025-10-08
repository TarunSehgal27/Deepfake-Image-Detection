import os
import streamlit as st
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Import model architecture from separate file (keep that file private)
try:
    from model_architecture import get_model
    MODEL_ARCHITECTURE_AVAILABLE = True
except ImportError:
    MODEL_ARCHITECTURE_AVAILABLE = False
    st.error("⚠️ model_architecture.py not found. Please create this file with your model definition.")

@st.cache_resource
def load_pytorch_model(model_path):
    """Load PyTorch model from .pt file with architecture from separate file"""
    try:
        if not MODEL_ARCHITECTURE_AVAILABLE:
            st.error("Model architecture file not found")
            return None, None
            
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Get model architecture from separate file
        model = get_model()
        
        # Load the state dict
        state_dict = torch.load(model_path, map_location=device)
        
        # Handle different save formats
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            model.load_state_dict(state_dict['state_dict'])
        else:
            model.load_state_dict(state_dict)
        
        model.eval()
        model.to(device)
        st.success("✅ Model loaded successfully")
        return model, device
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def preprocess_img(img):
    """Preprocess image for PyTorch model"""
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img_pil)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    return img_tensor

def predict_image(model, image, device):
    """Make prediction on image"""
    try:
        img_tensor = preprocess_img(image)
        img_tensor = img_tensor.to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted = torch.argmax(probabilities, 1).item()
            
            # Get confidence for the predicted class
            confidence = probabilities[0][predicted].item()
            
            # Get individual probabilities for real and fake
            real_prob = probabilities[0][0].item()
            fake_prob = probabilities[0][1].item()
            
        return predicted, confidence, real_prob, fake_prob
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None, None, None

# Page Configuration
img_path = os.path.join(BASE_DIR, "img", "page_icon.jfif")
img = Image.open(img_path)
st.set_page_config(
    page_title="Fake Image Detector", 
    page_icon=img, 
    initial_sidebar_state="expanded"
)

st.header("Deepfake Image Detection Tool")

# Sidebar
with st.sidebar:
    try:
        st.image("img/detector.png")
    except:
        st.write("🔍 Deepfake Detector")
    
    st.write("---")
    st.title("About")
    st.write(
        """This application represents the culmination of advanced research in computer vision and machine learning, specifically
        designed to stay ahead of evolving deepfake generation techniques. Our model continuously learns from new patterns and
        maintains effectiveness against emerging manipulation methods. The deepfake media circulating on the social media applications 
        can be checked using this tool.
        """
    )
    
    st.markdown(
        """
        ---
        Follow me on:
        
        Github → [@tarunsehgal27](https://github.com/tarunsehgal27)
        
        LinkedIn → [Tarun Sehgal](https://www.linkedin.com/in/tarunsehgal27)
        """
    )

# Model Loading
MODEL_PATH = os.path.join(BASE_DIR, "models", "deepfake_model.pt")

try:
    model, device = load_pytorch_model(MODEL_PATH)
    model_loaded = model is not None
except:
    model_loaded = False
    st.warning("⚠️ Model file not found. Please ensure 'model.pt' is in the correct directory.")

# File Upload
uploaded_file = st.file_uploader("Choose your file...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.write("✅ File Uploaded Successfully")
    
    # Read and display image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # Automatically analyze image when uploaded
    if model_loaded:
        with st.spinner("🔍 Analyzing image..."):
            result = predict_image(model, image, device)
            
            if result[0] is not None:
                predicted, confidence, real_prob, fake_prob = result
                
                # Create two columns for image and result
                col1, col2 = st.columns([1, 1], gap="large")
                
                with col1:
                    st.image(image, channels="BGR", width='stretch')
                
                with col2:
                    # Display professional results
                    if predicted == 0:  # Real Image (class 0)
                        st.markdown(
                            f"""
                            <span style="color: green; font-size: 30px;">Real</span> <br>
                            <span style="color: green; font-size: 30px;">Confidence: {confidence * 100:.1f}%</span>
                            """,
                            unsafe_allow_html=True
                        )
                    else:  # Deepfake (class 1)
                        st.markdown(
                            f"""
                            <span style="color: red; font-size: 30px;">⚠ Deepfake</span> <br>
                            <span style="color: red; font-size: 30px;">Confidence: {confidence * 100:.1f}%</span>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    # Additional info for low confidence
                    if confidence < 0.7:
                        st.warning("⚠️ Low confidence - results may be uncertain")
    else:
        st.error("❌ Model not loaded. Cannot perform analysis.")
        st.image(image, channels="BGR", caption="Uploaded Image", width='stretch')

st.write("---")

# Information Sections
st.subheader("Challenge We Address")
st.write(
    """Deepfake and AI generated images have become increasingly sophisticated, making it difficult to distinguish authentic
    content from manipulated media with the naked eye. Traditional detection methods fall short when faced with advanced generation
    techniques like GANs, Diffusion models etc., creating a need for more powerful detection solutions.
    """
)

st.subheader("Hybrid GAN-ResNet152 Architecture")
st.write(
    """This application is built on groundbreaking research that combines two powerful model architectures:
    GANs used for sophisticated deepfake creation.
    ResNet for feature extraction and analysis.
    """
)

st.subheader("Model Training Graph")
try:
    model_graph_path = os.path.join(BASE_DIR, "img", "model_accuracy.png")
    st.image(model_graph_path)
    st.markdown("**Model Accuracy: 88%**")
except:
    st.info("Model training graph not available")

# Footer
st.markdown(
    """
    <div style="margin-top: 50px; padding: 15px; text-align: center; border-top: 1px solid #ddd;">
        © <a href="https://github.com/tarunsehgal27" target="_blank">Tarun Sehgal</a> | <strong>Made with ❤️ </strong>
    </div>
    """,
    unsafe_allow_html=True
)