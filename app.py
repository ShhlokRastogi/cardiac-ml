import os
import sys

# Ensure src/ is on python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
from typing import List, Dict

from src.config import DEVICE, STAGE1_WEIGHTS_PATH, STAGE2_WEIGHTS_PATH, FEATURE_COLS
from src.predict import CardiacDiagnosisPipeline

app = FastAPI(
    title="Automated Cardiac MRI Segmentation & Pathology Diagnosis API",
    description="Production MLOps REST API powered by PyTorch Attention U-Net and Scikit-learn Random Forest Classifier.",
    version="1.0.0"
)

# Initialize pipeline engine
pipeline = CardiacDiagnosisPipeline()

class FeaturePayload(BaseModel):
    Height: float = 170.0
    Weight: float = 70.0
    BSA: float = 1.84
    RVEDV: float = 140.0
    RVESV: float = 50.0
    RVEF: float = 64.28
    RVEDVI: float = 76.08
    RVESVI: float = 27.17
    LVM_g: float = 120.0
    LVMI: float = 65.21
    LVEDV: float = 150.0
    LVESV: float = 60.0
    LVEF: float = 60.00
    LVEDVI: float = 81.52
    LVESVI: float = 32.60
    Max_MYO_Thickness_mm: float = 12.0

@app.get("/")
def root():
    return {
        "service": "Cardiac MRI Pathology Diagnosis API",
        "status": "online",
        "device": str(DEVICE),
        "docs_url": "/docs"
    }

@app.get("/health")
def health_check():
    stage1_status = os.path.exists(STAGE1_WEIGHTS_PATH)
    stage2_status = os.path.exists(STAGE2_WEIGHTS_PATH)
    return {
        "status": "healthy" if (stage1_status and stage2_status) else "degraded",
        "stage1_model_loaded": stage1_status,
        "stage2_model_loaded": stage2_status,
        "device": str(DEVICE)
    }

@app.post("/predict/pathology_from_features")
def predict_pathology_from_features(payload: FeaturePayload):
    if pipeline.stage2_classifier is None:
        raise HTTPException(status_code=500, detail="Stage 2 classifier model weights are not loaded.")
        
    feat_dict = payload.model_dump()
    x_feat = np.array([[feat_dict[col] for col in FEATURE_COLS]], dtype=np.float64)
    
    pred_class = pipeline.stage2_classifier.predict(x_feat)[0]
    probs = pipeline.stage2_classifier.predict_proba(x_feat)[0]
    confidence = float(np.max(probs) * 100.0)
    class_probabilities = {cls: float(p) for cls, p in zip(pipeline.stage2_classifier.classes_, probs)}
    
    return {
        "predicted_pathology": pred_class,
        "confidence_percentage": confidence,
        "class_probabilities": class_probabilities,
        "features_evaluated": feat_dict
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
