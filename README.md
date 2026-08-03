# 🫀 Automated Cardiac MRI Segmentation & Pathology Diagnosis Platform

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An end-to-end, production-grade MLOps medical AI system for 3D Cardiac Cine MRI segmentation and automated pathology classification (**DCM, HCM, MINF, NOR, RV**) evaluated on the ACDC dataset.

---

## 🏛️ Production System Architecture

```text
cardiac-ml/
├── src/                      # Clean Core Python Source Package
│   ├── __init__.py           # Package marker
│   ├── config.py             # Central path, device, & hyperparameter settings
│   ├── models.py             # Attention U-Net 2D/3D PyTorch architecture
│   ├── post_process.py       # 3D Morphological Connected Component Filter
│   ├── data_prep.py          # BSA, Volumetric Integration, & 16 Clinical Feature Calculators
│   ├── preprocess_dataset.py # Raw NIfTI Z-Score, 1.25mm Resampling, & 216x256 Cropping
│   ├── train.py              # Stage 1 (Att-UNet) and Stage 2 (Random Forest) Training
│   ├── evaluate.py           # Evaluation Metrics (Dice, R^2, Bland-Altman)
│   └── predict.py            # End-to-End Inference Engine Class
├── models/
│   ├── best_attention_unet_model.pth # Stage 1 Attention U-Net weights (~120MB - Git LFS)
│   └── acdc_disease_classifier.pkl   # Stage 2 Random Forest classifier weights (~2MB)
├── static/
│   └── index.html            # Web UI Interface (Served at GET /)
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py      # Automated Pytest Unit Test Suite
├── .github/workflows/
│   └── ci.yaml               # GitHub Actions CI Workflow
├── app.py                    # FastAPI REST Server & Image Processing Engine
├── Dockerfile                # Production Container Recipe (Lightweight CPU PyTorch)
├── requirements.txt          # Production Dependencies
└── README.md                 # Complete Documentation
```

---

## 📊 Performance Benchmarks

* **Stage 1 (Attention U-Net 3D Segmentation)**:
  * **Mean Dice Overlap**: **`89.13%`** ($LV: 92.56\%, RV: 88.70\%, MYO: 86.12\%$)
  * **Volume Correlation ($R^2$)**: $LV = 0.9902, RV = 0.9658, MYO = 0.9321$
* **Stage 2 (Random Forest Pathology Classifier)**:
  * **5-Fold Cross-Validation Accuracy**: **`94.00%`**
  * **Unseen Test Set Accuracy**: **`84.00%`** (42 / 50 unseen test patients correctly diagnosed)

---

## 🌐 Production API Endpoints

| Endpoint | Method | Input | Description |
| :--- | :---: | :--- | :--- |
| **`/`** | `GET` | — | Interactive Web UI Dashboard |
| **`/docs`** | `GET` | — | Swagger API Interactive Documentation |
| **`/predict/from_raw_nifti`** | `POST` | `.nii` / `.nii.gz` ED & ES files | End-to-end raw MRI scan segmentation & pathology diagnosis |

---

## 🚀 Quick Start

### 1. Local Installation

```bash
git clone https://github.com/ShhlokRastogi/cardiac-ml.git
cd cardiac-ml
pip install -r requirements.txt
```

### 2. Run Automated Pytest Suite

```bash
pytest tests/
```

### 3. Launch Web Server & UI

```bash
python -m uvicorn app:app --reload --port 8000
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your web browser.

---

## 🐳 Docker Deployment

To build and run the lightweight container locally (~350 MB image):

```bash
docker build -t cardiac-ml:latest .
docker run -p 8000:8000 cardiac-ml:latest
```

---

## 📜 License
This project is licensed under the MIT License.
