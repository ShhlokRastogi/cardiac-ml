# 🫀 Automated Cardiac MRI Segmentation & Pathology Diagnosis Platform

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An end-to-end, production-grade MLOps medical AI system for 3D Cardiac Cine MRI segmentation and automated pathology classification (**DCM, HCM, MINF, NOR, RV**), restructured into decoupled `/backend` and `/frontend` services for independent deployment.

---

## 🏛️ Decoupled Repository Architecture

```text
cardiac-ml/
├── backend/                      # FastAPI Backend & PyTorch MLOps Engine
│   ├── app.py                    # REST API application (with CORS middleware)
│   ├── Dockerfile                # Backend production container recipe
│   ├── requirements.txt          # Python package dependencies
│   ├── .env.example              # Backend environment template
│   ├── src/                      # Core MLOps Python modules
│   ├── models/                   # PyTorch & Random Forest weights
│   └── tests/                    # Pytest unit tests
├── frontend/                     # Standalone Web UI Client
│   ├── index.html                # Single-page Web UI Dashboard
│   ├── package.json              # Local dev server scripts & dependencies
│   └── .env.example              # Frontend environment template
├── .github/workflows/
│   └── ci.yaml                   # Automated GitHub Actions CI workflow
├── DEPLOYMENT.md                 # Multi-service deployment guide (Render/Vercel/Railway)
└── README.md                     # Repository Overview
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

## 🚀 Quick Start (Local Development)

### 1. Run Backend Service (Terminal 1)
```bash
cd backend
pip install -r requirements.txt python-multipart
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
* **API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)

### 2. Run Frontend Client (Terminal 2)
```bash
cd frontend
npm start
```
* **Web UI Dashboard**: [http://localhost:3000](http://localhost:3000)

---

## 📖 Independent Deployment Guide
For detailed instructions on configuring environment variables (`FRONTEND_URL`, `VITE_API_URL`) and deploying backend and frontend as separate cloud services from this repository, see **[DEPLOYMENT.md](DEPLOYMENT.md)**.

---

## 📜 License
This project is licensed under the MIT License.
