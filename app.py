import os
import sys
import io
import tempfile

# Ensure root directory and /app are at the head of python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.getcwd())
if os.path.exists("/app"):
    sys.path.insert(0, "/app")

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

try:
    from src.config import DEVICE, STAGE1_WEIGHTS_PATH, STAGE2_WEIGHTS_PATH, FEATURE_COLS
    from src.predict import CardiacDiagnosisPipeline
    from src.preprocess_dataset import preprocess_slice_exact
except ImportError:
    from config import DEVICE, STAGE1_WEIGHTS_PATH, STAGE2_WEIGHTS_PATH, FEATURE_COLS
    from predict import CardiacDiagnosisPipeline
    from preprocess_dataset import preprocess_slice_exact

app = FastAPI(
    title="Automated Cardiac MRI Segmentation & Pathology Diagnosis API",
    description="Production MLOps REST API powered by PyTorch Attention U-Net and Scikit-learn Random Forest Classifier. Supports raw NIfTI (.nii/.nii.gz) and Numpy (.npy) image uploads.",
    version="1.2.2"
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
        "device": str(DEVICE),
        "supports_raw_nifti_uploads": HAS_NIBABEL
    }

@app.post("/predict/from_raw_nifti")
async def predict_from_raw_nifti(
    patient_id: str = Form("patient_raw"),
    height_cm: float = Form(170.0),
    weight_kg: float = Form(70.0),
    ed_nii_file: UploadFile = File(..., description="Raw NIfTI (.nii or .nii.gz) for End-Diastole frame"),
    es_nii_file: UploadFile = File(..., description="Raw NIfTI (.nii or .nii.gz) for End-Systole frame")
):
    """
    Directly accepts RAW MRI NIfTI scans (.nii / .nii.gz) for ED and ES cardiac frames.
    Performs on-the-fly intensity normalization, resampling, Attention U-Net 3D segmentation,
    clinical feature calculation, and automated disease diagnosis.
    """
    if not HAS_NIBABEL:
        raise HTTPException(status_code=500, detail="nibabel library is required for raw NIfTI parsing.")
        
    try:
        # Detect exact file extensions (.nii vs .nii.gz)
        ext_ed = ".nii.gz" if ed_nii_file.filename.endswith(".nii.gz") else ".nii"
        ext_es = ".nii.gz" if es_nii_file.filename.endswith(".nii.gz") else ".nii"
        
        # Save temporary files with correct extensions
        with tempfile.NamedTemporaryFile(suffix=ext_ed, delete=False) as tmp_ed:
            tmp_ed.write(await ed_nii_file.read())
            tmp_ed_path = tmp_ed.name
            
        with tempfile.NamedTemporaryFile(suffix=ext_es, delete=False) as tmp_es:
            tmp_es.write(await es_nii_file.read())
            tmp_es_path = tmp_es.name
            
        # Parse NIfTI files
        nii_ed = nib.load(tmp_ed_path)
        nii_es = nib.load(tmp_es_path)
        
        ed_raw = nii_ed.get_fdata()
        es_raw = nii_es.get_fdata()
        
        zooms_ed = nii_ed.header.get_zooms()[:2]
        zooms_es = nii_es.header.get_zooms()[:2]
        
        # Clean up temporary files
        os.remove(tmp_ed_path)
        os.remove(tmp_es_path)
        
        # Preprocess slices (Z-Score + Resample + Crop)
        ed_slices = [preprocess_slice_exact(ed_raw[:, :, s], current_spacing=zooms_ed)[0] for s in range(ed_raw.shape[2])]
        es_slices = [preprocess_slice_exact(es_raw[:, :, s], current_spacing=zooms_es)[0] for s in range(es_raw.shape[2])]
        
        ed_vol = np.stack(ed_slices, axis=2).astype(np.float32)
        es_vol = np.stack(es_slices, axis=2).astype(np.float32)
        
        info_dict = {"Height": str(height_cm), "Weight": str(weight_kg), "Group": "Unknown"}
        
        results = pipeline.predict_patient_end_to_end(patient_id, info_dict, ed_vol, es_vol)
        
        return {
            "patient_id": results["Patient_ID"],
            "raw_files_processed": {
                "ED_file": ed_nii_file.filename,
                "ES_file": es_nii_file.filename
            },
            "predicted_diagnosis": results["Predicted_Diagnosis"],
            "confidence_percentage": results["Confidence_Percentage"],
            "class_probabilities": results["Class_Probabilities"],
            "clinical_features": results["Clinical_Features"]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process raw NIfTI files: {str(e)}")

@app.post("/predict/end_to_end_from_volumes")
async def predict_end_to_end_from_volumes(
    patient_id: str = Form("patient_test"),
    height_cm: float = Form(170.0),
    weight_kg: float = Form(70.0),
    ed_file: UploadFile = File(...),
    es_file: UploadFile = File(...)
):
    """
    Accepts 3D .npy volume image files + height & weight.
    Executes full pipeline (Segmentation -> Feature Extraction -> Disease Classification).
    """
    try:
        ed_contents = await ed_file.read()
        es_contents = await es_file.read()
        
        ed_vol = np.load(io.BytesIO(ed_contents)).astype(np.float32)
        es_vol = np.load(io.BytesIO(es_contents)).astype(np.float32)
        
        if ed_vol.ndim == 2: ed_vol = np.expand_dims(ed_vol, axis=2)
        if es_vol.ndim == 2: es_vol = np.expand_dims(es_vol, axis=2)
        
        info_dict = {"Height": str(height_cm), "Weight": str(weight_kg), "Group": "Unknown"}
        
        results = pipeline.predict_patient_end_to_end(patient_id, info_dict, ed_vol, es_vol)
        
        return {
            "patient_id": results["Patient_ID"],
            "predicted_diagnosis": results["Predicted_Diagnosis"],
            "confidence_percentage": results["Confidence_Percentage"],
            "class_probabilities": results["Class_Probabilities"],
            "clinical_features": results["Clinical_Features"]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"End-to-end image processing error: {str(e)}")

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
