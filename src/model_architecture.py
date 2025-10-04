"""
Model Architecture File
Keep this file private and don't share it publicly
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

def get_model():
    """Returns an instance of the EfficientNet-B0 model for deepfake detection"""
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)
    
    # Replace classifier for binary classification (Real vs Fake)
    in_features = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.4),
        torch.nn.Linear(in_features, 2)
    )
    
    return model