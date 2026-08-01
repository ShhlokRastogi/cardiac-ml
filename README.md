# Production MLOps Pipeline for Cardiac MRI Segmentation & Pathology Diagnosis

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An end-to-end medical AI pipeline for automated 3D Cardiac Cine MRI segmentation and pathology classification (DCM, HCM, MINF, NOR, RV) evaluated on the ACDC dataset.

---

## 🏗️ Repository Architecture

```text
mlops-project/
├── data/
│   ├── raw/                  # Reference pointer to raw ACDC NIfTI dataset
│   └── processed/            # Reference pointer to preprocessed numpy 2D/3D stacks
├── src/
│   ├── __init__.py           # Package marker
│   ├── config.py             # Central path & hyperparameter configuration
│   ├── models.py             # Attention U-Net PyTorch architecture definition
│   ├── post_process.py       # 3D Connected Component morphological noise filter
│   ├── data_prep.py          # NIfTI parsing, BSA, volume, & clinical feature extraction
│   ├── train.py              # Stage 1 (Att-UNet) and Stage 2 (Random Forest) training modules
│   ├── evaluate.py           # Metrics calculation (Dice, R^2, Bland-Altman, F1-score)
│   └── predict.py            # End-to-end inference engine class (MRI -> Pathology)
├── models/
│   ├── best_attention_unet_model.pth     # Stage 1 Attention U-Net weights (~120MB)
│   └── acdc_disease_classifier.pkl       # Stage 2 Random Forest classifier weights (~2MB)
├── notebooks/                # Experimentation & diagnostic plotting notebooks
├── app.py                    # FastAPI web server serving live REST API endpoints
├── Dockerfile                # Docker container recipe for PyTorch + FastAPI
├── requirements.txt          # Pinned python package dependencies
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py      # Automated pytest unit test suite
├── .github/workflows/
│   └── ci.yaml               # GitHub Actions CI workflow script
└── README.md                 # Complete project documentation
```

---

## 📊 Performance Benchmarks

* **Stage 1 (Attention U-Net 3D Segmentation)**:
  * **Mean Dice Overlap**: **`89.13%`** (LV: 92.56%, RV: 88.70%, MYO: 86.12%)
  * **Volume Correlation ($R^2$)**: $LV = 0.9902, RV = 0.9658, MYO = 0.9321$
* **Stage 2 (Random Forest Pathology Classifier)**:
  * **5-Fold Cross-Validation Accuracy**: **`94.00%`**
  * **Unseen Test Set Accuracy**: **`84.00%`** (42 / 50 unseen test patients correctly diagnosed end-to-end)

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/your-username/cardiac-mlops-acdc.git
cd mlops-project
pip install -r requirements.txt
```

### 2. Run Pytest Suite

```bash
pytest tests/
```

### 3. Launch FastAPI REST Server

```bash
uvicorn app:app --reload --port 8000
```
Open your browser and navigate to **`http://127.0.0.1:8000/docs`** to access the interactive Swagger API documentation.

---

## 🐳 Docker Deployment

To build and run the production container:

```bash
docker build -t cardiac-mlops:latest .
docker run -p 8000:8000 cardiac-mlops:latest
```

---

## 📜 License
This project is licensed under the MIT License.
