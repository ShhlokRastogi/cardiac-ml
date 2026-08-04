import os
import sys
import tempfile
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware

# Ensure root directory is first on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

from src.predict import CardiacDiagnosisPipeline
from src.preprocess_dataset import preprocess_slice_exact

app = FastAPI(
    title="Automated Cardiac MRI Segmentation & Pathology Diagnosis API",
    description="Production MLOps REST API backend for raw NIfTI (.nii / .nii.gz) MRI scan segmentation & pathology diagnosis.",
    version="2.1.0"
)

# Read allowed frontend origin from environment variable
frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
allowed_origins = [
    frontend_url,
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline engine
pipeline = CardiacDiagnosisPipeline()


@app.get("/")
def root():
    return {
        "service": "Cardiac MRI Pathology Diagnosis API Backend",
        "status": "online",
        "docs_url": "/docs",
        "prediction_endpoint": "/predict/from_raw_nifti"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "has_nibabel": HAS_NIBABEL,
        "stage1_model_loaded": pipeline.stage1_model is not None,
        "stage2_model_loaded": pipeline.stage2_classifier is not None
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
    Accepts RAW MRI NIfTI scans (.nii / .nii.gz) for ED and ES cardiac frames.
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


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
