"""
backend/main.py
===============
FastAPI service for the Student Performance Predictor.
Loads the trained GradientBoosting model at startup and exposes two endpoints:
  GET  /health   → liveness check
  POST /predict  → predict final grade (0–100 %)

Run locally:
    uvicorn backend.main:app --port 8001 --reload

Deployed on Render via render.yaml (auto-starts on push to main).
"""

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

# ---------------------------------------------------------------------------
# Path resolution — works locally AND on Render
# ---------------------------------------------------------------------------
# This file lives at  <project_root>/backend/main.py
# Models live at      <project_root>/models/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"

# Make sure inference.py (at project root) is importable
sys.path.insert(0, str(PROJECT_ROOT))
from inference import load_production_model, predict_single   # noqa: E402

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Student Performance Predictor API",
    description="Predict a student's final course grade using ML.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Load model at startup
# ---------------------------------------------------------------------------
try:
    model, pipe, feature_cols = load_production_model(
        model_name="gradient_boosting",
        models_dir=str(MODELS_DIR),
    )
    print(f"[OK] API ready. Model loaded from: {MODELS_DIR}")
except Exception as exc:
    print(f"[ERROR] Model failed to load: {exc}")
    model, pipe, feature_cols = None, None, None


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class PredictionRequest(BaseModel):
    attendance_rate:          float           = Field(..., ge=0, le=100,  description="Attendance rate (0–100 %)")
    weekly_study_hours:       float           = Field(..., ge=0, le=168,  description="Weekly study hours")
    past_exam_scores:         float           = Field(..., ge=0, le=100,  description="Average past exam score (0–100 %)")
    homework_completion_rate: Optional[float] = Field(None, ge=0, le=100, description="Homework completion rate – optional (0–100 %)")


class PredictionResponse(BaseModel):
    predicted_grade: float
    confidence_note: str
    input_features:  dict


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Ops"])
def health_check():
    """Liveness probe – returns 200 when the model is loaded."""
    if model is None:
        return {"status": "error", "message": "Model failed to load."}
    return {"status": "ok", "message": "Model is loaded and ready."}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(req: PredictionRequest):
    """Predict a student's final grade percentage (0–100)."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model is currently unavailable.")

    kwargs = {
        "attendance_rate":    req.attendance_rate,
        "weekly_study_hours": req.weekly_study_hours,
        "past_exam_scores":   req.past_exam_scores,
    }
    if req.homework_completion_rate is not None:
        kwargs["homework_completion_rate"] = req.homework_completion_rate

    try:
        result = predict_single(model, pipe, feature_cols, **kwargs)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")


# ---------------------------------------------------------------------------
# Local dev entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8001, reload=True)
