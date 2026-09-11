import os
import sys
import gc
import tempfile
import time
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query
from fastapi.middleware.cors import CORSMiddleware

# Restrict PyTorch thread memory allocation on cloud free tiers (Render 512MB RAM limit)
torch.set_num_threads(1)
if hasattr(torch, "set_num_interop_threads"):
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass

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
from src.observability import (
    LatencyTimer,
    DataDriftDetector,
    trace_logger,
    metrics_collector
)
from src.evals import EvaluatorEngine

app = FastAPI(
    title="Automated Cardiac MRI Segmentation & Pathology Diagnosis API",
    description="Production MLOps REST API backend for raw NIfTI (.nii / .nii.gz) MRI scan segmentation & pathology diagnosis with full Observability & Evals.",
    version="2.3.0"
)

# Read allowed frontend origin from environment variable
env_frontend = os.getenv("FRONTEND_URL") or os.getenv("frontend_url") or "http://localhost:3000"
env_frontend_clean = env_frontend.rstrip("/")

allowed_origins = [
    env_frontend_clean,
    f"{env_frontend_clean}/",
    "https://cardiac-ml.vercel.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline engine (Lazy loads PyTorch weights on demand)
pipeline = CardiacDiagnosisPipeline()


@app.get("/")
def root():
    return {
        "service": "Cardiac MRI Pathology Diagnosis API Backend",
        "status": "online",
        "version": "2.3.0",
        "docs_url": "/docs",
        "endpoints": {
            "prediction": "/predict/from_raw_nifti",
            "metrics": "/metrics",
            "traces": "/observability/traces",
            "evals_report": "/evals/report"
        }
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "has_nibabel": HAS_NIBABEL,
        "stage1_weights_exist": os.path.exists(pipeline.stage1_weights),
        "stage2_weights_exist": os.path.exists(pipeline.stage2_weights)
    }


@app.get("/metrics")
def get_prometheus_metrics():
    """
    Returns aggregated Prometheus/JSON observability metrics:
    - Total prediction count
    - Average total pipeline latency (ms)
    - Sub-phase latency breakdown (ms)
    - Diagnosis class distribution
    - Total data drift alert count
    """
    return metrics_collector.get_summary_metrics()


@app.get("/observability/traces")
def get_recent_traces(limit: int = Query(20, ge=1, le=100)):
    """Returns the most recent structured JSONL inference traces."""
    return trace_logger.read_recent_traces(limit=limit)


@app.get("/evals/report")
def get_evaluation_report():
    """Returns automated evaluation benchmark metrics and data drift status."""
    return EvaluatorEngine.generate_full_eval_report()


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
    Performs full sub-phase latency tracking, intensity normalization, resampling,
    Attention U-Net 3D segmentation, clinical feature calculation, data drift checks,
    and automated disease diagnosis.
    """
    if not HAS_NIBABEL:
        raise HTTPException(status_code=500, detail="nibabel library is required for raw NIfTI parsing.")

    timer = LatencyTimer()

    try:
        # Phase 1: NIfTI I/O
        t0 = time.perf_counter()
        ext_ed = ".nii.gz" if ed_nii_file.filename.endswith(".nii.gz") else ".nii"
        ext_es = ".nii.gz" if es_nii_file.filename.endswith(".nii.gz") else ".nii"

        with tempfile.NamedTemporaryFile(suffix=ext_ed, delete=False) as tmp_ed:
            tmp_ed.write(await ed_nii_file.read())
            tmp_ed_path = tmp_ed.name

        with tempfile.NamedTemporaryFile(suffix=ext_es, delete=False) as tmp_es:
            tmp_es.write(await es_nii_file.read())
            tmp_es_path = tmp_es.name

        nii_ed = nib.load(tmp_ed_path)
        nii_es = nib.load(tmp_es_path)

        ed_raw = nii_ed.get_fdata()
        es_raw = nii_es.get_fdata()

        zooms_ed = nii_ed.header.get_zooms()[:2]
        zooms_es = nii_es.header.get_zooms()[:2]

        os.remove(tmp_ed_path)
        os.remove(tmp_es_path)
        timer.measure("nifti_io_ms", t0)

        # Phase 2: Slice Preprocessing (Normalization + Spline Resampling + Crop)
        t1 = time.perf_counter()
        ed_slices = [preprocess_slice_exact(ed_raw[:, :, s], current_spacing=zooms_ed)[0] for s in range(ed_raw.shape[2])]
        es_slices = [preprocess_slice_exact(es_raw[:, :, s], current_spacing=zooms_es)[0] for s in range(es_raw.shape[2])]

        del ed_raw, es_raw
        gc.collect()

        ed_vol = np.stack(ed_slices, axis=2).astype(np.float32)
        es_vol = np.stack(es_slices, axis=2).astype(np.float32)

        del ed_slices, es_slices
        gc.collect()
        timer.measure("preprocessing_ms", t1)

        # Phase 3: Stage 1 Deep Learning Segmentation (Attention U-Net)
        t2 = time.perf_counter()
        ed_mask_pred = pipeline.predict_3d_volume(ed_vol)
        es_mask_pred = pipeline.predict_3d_volume(es_vol)

        del ed_vol, es_vol
        gc.collect()
        timer.measure("stage1_segmentation_ms", t2)

        # Phase 4: Post-Processing & 16 Clinical Biometrics Calculation
        t3 = time.perf_counter()
        info_dict = {"Height": str(height_cm), "Weight": str(weight_kg), "Group": "Unknown"}
        features = pipeline.extract_biometrics(patient_id, info_dict, ed_mask_pred, es_mask_pred)

        del ed_mask_pred, es_mask_pred
        gc.collect()
        timer.measure("postprocessing_biometrics_ms", t3)

        # Phase 5: Stage 2 Random Forest Disease Classification
        t4 = time.perf_counter()
        prediction, confidence, probs = pipeline.classify_disease(features)
        timer.measure("stage2_classification_ms", t4)

        total_ms = timer.total()
        latency_breakdown = {
            "nifti_io_ms": timer.latencies.get("nifti_io_ms", 0.0),
            "preprocessing_ms": timer.latencies.get("preprocessing_ms", 0.0),
            "stage1_segmentation_ms": timer.latencies.get("stage1_segmentation_ms", 0.0),
            "postprocessing_biometrics_ms": timer.latencies.get("postprocessing_biometrics_ms", 0.0),
            "stage2_classification_ms": timer.latencies.get("stage2_classification_ms", 0.0),
            "total_pipeline_ms": total_ms
        }

        # Data Drift & Out-of-Bounds Biometric Check
        drift_check = DataDriftDetector.check_biometric_drift(features)

        # Record metrics & write JSONL trace
        metrics_collector.record_inference(
            predicted_class=prediction,
            latency_breakdown=latency_breakdown,
            has_drift=drift_check["has_data_drift"]
        )

        trace_record = {
            "trace_id": str(time.time_ns()),
            "timestamp": str(np.datetime64("now")),
            "patient_id": patient_id,
            "predicted_diagnosis": prediction,
            "confidence_percentage": round(confidence, 2),
            "latency_breakdown_ms": latency_breakdown,
            "data_drift": drift_check
        }
        trace_logger.log_trace(trace_record)

        gc.collect()

        return {
            "patient_id": patient_id,
            "predicted_diagnosis": prediction,
            "confidence_percentage": confidence,
            "class_probabilities": probs,
            "clinical_features": features,
            "latency_breakdown_ms": latency_breakdown,
            "data_drift_monitoring": drift_check
        }

    except Exception as e:
        gc.collect()
        raise HTTPException(status_code=400, detail=f"Failed to process raw NIfTI files: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
