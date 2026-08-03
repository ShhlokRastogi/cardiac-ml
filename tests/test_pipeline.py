import os
import torch
import numpy as np
from src.config import INPUT_SHAPE_2D, DEVICE
from src.models import AttentionUNet
from src.data_prep import calculate_bsa, calculate_volume_ml

def test_model_architecture():
    """Verify Attention U-Net input and output tensor shapes."""
    model = AttentionUNet(n_channels=1, n_classes=4, bilinear=False).to(DEVICE)
    dummy_input = torch.randn(1, 1, 216, 256).to(DEVICE)
    output = model(dummy_input)
    assert output.shape == (1, 4, 216, 256), f"Expected shape (1, 4, 216, 256), got {output.shape}"

def test_clinical_feature_calculations():
    """Verify BSA and volumetric feature calculations."""
    bsa = calculate_bsa(height_cm=175.0, weight_kg=75.0)
    assert round(bsa, 2) == 1.90, f"Expected BSA ~1.90, got {bsa}"
    
    dummy_mask_3d = np.zeros((10, 10, 5), dtype=np.uint8)
    dummy_mask_3d[2:8, 2:8, :] = 3 # class 3 (LV)
    vol_ml = calculate_volume_ml(dummy_mask_3d, class_idx=3)
    assert vol_ml > 0.0, "Calculated volume should be greater than zero"

def test_fastapi_app_import():
    """Verify FastAPI application and pipeline load without syntax/import errors."""
    from app import app
    assert app.title == "Automated Cardiac MRI Segmentation & Pathology Diagnosis API"
