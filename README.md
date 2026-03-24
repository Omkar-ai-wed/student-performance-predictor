# 🎓 Student Performance Predictor

An AI-powered web application that predicts a student's final exam grade based on key academic behaviour inputs. Built with a **FastAPI** backend, a **React + Tailwind CSS** frontend, and a **Gradient Boosting** ML model — deployed via **Render** (API) and **GitHub Pages** (UI).

---

## ✨ Features

- 📊 Predicts final course grade (0–100 %) in real time
- 🤖 Gradient Boosting model trained on synthetic student data
- ⚡ FastAPI REST API with automatic Swagger docs (`/docs`)
- 🎨 Glassmorphic React UI with Tailwind CSS
- 🚀 CI/CD via GitHub Actions → auto-deploys frontend to GitHub Pages on every push to `main`

---

## 🏗️ Project Structure

```
student_performance_predictor/
├── backend/
│   ├── main.py              # FastAPI app (GET /health, POST /predict)
│   └── requirements.txt     # Backend-only deps (fastapi, uvicorn, …)
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main React component
│   │   └── index.css        # Glassmorphic styles
│   ├── package.json
│   └── vite.config.js
├── models/
│   ├── gradient_boosting.joblib      # Trained ML model
│   ├── preprocessing_pipeline.joblib # Feature scaler / encoder
│   └── feature_cols.json            # Column metadata
├── data/
│   └── schema.json          # Dataset schema
├── .github/
│   └── workflows/
│       └── deploy.yml       # GitHub Actions → GitHub Pages
├── data_generator.py        # Synthetic dataset generator
├── model_training.py        # Train & save the model
├── preprocessing.py         # Feature engineering pipeline
├── inference.py             # load_production_model / predict_single helpers
├── evaluation.py            # Model evaluation metrics
├── hyperparameter_tuning.py # GridSearch / RandomSearch helpers
├── main.py                  # Standalone CLI predictor
├── render.yaml              # Render deployment config (backend)
└── requirements.txt         # Root-level Python deps
```

---

## 🚀 Quick Start

### 1 — Clone the repo

```bash
git clone https://github.com/Omkar-ai-wed/student-performance-predictor.git
cd student-performance-predictor
```

### 2 — Backend (FastAPI)

```bash
# Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -r backend/requirements.txt
uvicorn backend.main:app --port 8001 --reload
```

API available at → `http://localhost:8001`  
Interactive docs → `http://localhost:8001/docs`

### 3 — Frontend (React + Vite)

```bash
cd frontend
npm install
npm run dev
```

UI available at → `http://localhost:5173`

> **Tip:** set `VITE_API_URL=http://localhost:8001` in `frontend/.env.development.local` to point the UI at your local backend.

---

## 🔌 API Reference

### `GET /health`

Liveness probe — returns `200 OK` when the model is loaded.

```json
{ "status": "ok", "message": "Model is loaded and ready." }
```

### `POST /predict`

Predict a student's final grade.

**Request body:**

| Field | Type | Required | Range | Description |
|---|---|---|---|---|
| `attendance_rate` | float | ✅ | 0 – 100 | % of classes attended |
| `weekly_study_hours` | float | ✅ | 0 – 168 | Hours studied per week |
| `past_exam_scores` | float | ✅ | 0 – 100 | Average score on past exams |
| `homework_completion_rate` | float | ❌ | 0 – 100 | % homework submitted |

**Example:**

```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"attendance_rate": 85, "weekly_study_hours": 12, "past_exam_scores": 78}'
```

**Response:**

```json
{
  "predicted_grade": 81.4,
  "confidence_note": "Gradient Boosting prediction",
  "input_features": { ... }
}
```

---

## 🧠 ML Pipeline

| Step | Tool | Detail |
|---|---|---|
| Data generation | `data_generator.py` | Synthetic students dataset |
| Preprocessing | `preprocessing.py` | Scaling, imputation, encoding |
| Training | `model_training.py` | `GradientBoostingRegressor` |
| Tuning | `hyperparameter_tuning.py` | GridSearch / RandomSearch |
| Evaluation | `evaluation.py` | MAE, RMSE, R² metrics |
| Inference | `inference.py` | `load_production_model()` + `predict_single()` |

---

## ☁️ Deployment

| Layer | Platform | Trigger |
|---|---|---|
| **Backend API** | [Render](https://render.com) | Push to `main` (via `render.yaml`) |
| **Frontend UI** | GitHub Pages | Push to `main` (via GitHub Actions) |

### GitHub Pages setup

The GitHub Actions workflow (`.github/workflows/deploy.yml`) automatically builds the React app and deploys it to the `gh-pages` branch on every push to `main`.

Add your Render backend URL as a **GitHub secret** named `VITE_API_URL` so the frontend can reach the API in production:

```
Settings → Secrets and variables → Actions → New repository secret
Name:  VITE_API_URL
Value: https://<your-render-service>.onrender.com
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML | scikit-learn, NumPy, pandas, joblib |
| Backend | FastAPI, Uvicorn, Pydantic |
| Frontend | React 19, Vite, Tailwind CSS 3 |
| CI/CD | GitHub Actions, Render |

---

## 📄 License

MIT — feel free to use and adapt.
