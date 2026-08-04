import os
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data Directories
RAW_TRAIN_DB_DIR = os.path.join(BASE_DIR, "data", "raw", "database", "training")
RAW_TEST_DB_DIR = os.path.join(BASE_DIR, "data", "raw", "database", "testing")

PREPROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed", "preprocessed data")
PREPROCESSED_IMG_TRAIN = os.path.join(PREPROCESSED_DIR, "images", "train")
PREPROCESSED_MASK_TRAIN = os.path.join(PREPROCESSED_DIR, "masks", "train")
PREPROCESSED_IMG_TEST = os.path.join(PREPROCESSED_DIR, "images", "test")
PREPROCESSED_MASK_TEST = os.path.join(PREPROCESSED_DIR, "masks", "test")

# Weights Paths (Supports both naming conventions)
MODELS_DIR = os.path.join(BASE_DIR, "models")
STAGE1_WEIGHTS_PATH = os.path.join(MODELS_DIR, "best_attention_unet_model.pth")

stage2_att = os.path.join(MODELS_DIR, "acdc_disease_classifier_att_pred.pkl")
stage2_std = os.path.join(MODELS_DIR, "acdc_disease_classifier.pkl")
STAGE2_WEIGHTS_PATH = stage2_att if os.path.exists(stage2_att) else stage2_std

# Device Settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Physical Spacing & Dimensions
VOXEL_SPACING = (1.25, 1.25, 10.0)  # (in-plane x, in-plane y, slice thickness z mm)
INPUT_SHAPE_2D = (216, 256)

# Disease Target Classes
DISEASE_CLASSES = ["DCM", "HCM", "MINF", "NOR", "RV"]

# Clinical Feature Names (16 Features)
FEATURE_COLS = [
    "Height", "Weight", "BSA",
    "RVEDV", "RVESV", "RVEF", "RVEDVI", "RVESVI",
    "LVM_g", "LVMI", "LVEDV", "LVESV", "LVEF", "LVEDVI", "LVESVI",
    "Max_MYO_Thickness_mm"
]
