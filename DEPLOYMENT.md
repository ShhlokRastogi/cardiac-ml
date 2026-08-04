# 🚀 Decoupled Deployment Guide: Backend & Frontend

This document explains how to run, configure, and deploy the **Cardiac AI Pathology Diagnosis Platform** as independent services (`/backend` and `/frontend`) from a single monorepo.

---

## 🏛️ Repository Folder Structure

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
└── DEPLOYMENT.md                 # Deployment & Environment Reference
```

---

## ⚙️ Environment Variables Reference

### 1. Backend (`/backend/.env`)

| Variable Name | Required | Default Value | Description |
| :--- | :---: | :--- | :--- |
| `FRONTEND_URL` | Optional | `http://localhost:3000` | Allowed origin URL for CORS requests from the frontend client. |
| `PORT` | Optional | `8000` | HTTP port on which the FastAPI server listens. |

### 2. Frontend (`/frontend/.env`)

| Variable Name | Required | Default Value | Description |
| :--- | :---: | :--- | :--- |
| `VITE_API_URL` | Optional | `http://localhost:8000` | Base API URL pointing to the deployed backend service. |

---

## 💻 Local Development Setup

To run both services locally on different ports (`http://localhost:8000` for Backend, `http://localhost:3000` for Frontend):

### Terminal 1: Backend Service
```bash
cd backend
pip install -r requirements.txt python-multipart
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
* **Backend API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

### Terminal 2: Frontend Service
```bash
cd frontend
npm start
```
* **Frontend Web UI**: [http://localhost:3000](http://localhost:3000)

---

## ☁️ Independent Cloud Deployment (Render / Vercel / Netlify)

You can deploy both services from the **same GitHub repository** (`https://github.com/ShhlokRastogi/cardiac-ml`) by configuring the **Root Directory** setting on your hosting platforms:

### 1. Deploying Backend (e.g. Render / Railway / AWS App Runner)
- **Repository URL**: `https://github.com/ShhlokRastogi/cardiac-ml`
- **Environment**: `Docker` or `Python 3.11`
- **Root Directory**: **`backend`**
- **Build Command**: `pip install -r requirements.txt` (or Docker build)
- **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
- **Environment Variables**:
  - `FRONTEND_URL` = `https://your-frontend-domain.vercel.app`

### 2. Deploying Frontend (e.g. Vercel / Netlify / Cloudflare Pages)
- **Repository URL**: `https://github.com/ShhlokRastogi/cardiac-ml`
- **Root Directory**: **`frontend`**
- **Build Command**: Leave empty or `npm run build`
- **Output Directory**: `.` (Root of `frontend`)
- **Environment Variables**:
  - `VITE_API_URL` = `https://your-backend-service.onrender.com`
