"""
Day 34: Deploying the Model Behind a Minimal FastAPI Endpoint

Day 33 saved a fitted pipeline to disk with joblib. Today that saved file
finally gets used for what persistence exists to enable: serving live
predictions over HTTP, without ever rerunning the training code.

The pipeline is loaded exactly ONCE, at application startup — not on every
request — using FastAPI's `lifespan` context manager (the current recommended
way to run startup/shutdown code, replacing the older `@app.on_event`
decorator). A Pydantic model defines and validates the shape of every
incoming request automatically, before the handler function ever runs.

Run for real traffic with:
    uvicorn day34_fastapi_deployment:app --reload
Then POST JSON to http://127.0.0.1:8000/predict

Running this file directly (`python day34_fastapi_deployment.py`) instead
exercises the app in-process with FastAPI's TestClient — no real network
socket, no separate terminal needed — which is what the verification block
at the bottom does.
"""

import os
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Day 33's pipeline includes FunctionTransformer(add_family_size). joblib/pickle
# stores a FUNCTION REFERENCE (module + name), not the function's body — so
# unpickling here fails with "Can't get attribute 'add_family_size'" unless the
# exact same function is importable from the exact same module path it was
# saved under. This import exists purely to make that resolvable; it is never
# called directly in this file.
from day33_model_persistence import add_family_size  # noqa: F401

MODEL_PATH = os.environ.get("MODEL_PATH", "titanic_pipeline.joblib")


# ---------------------------------------------------------------------------
# Request / response schemas — Pydantic validates every request automatically
# ---------------------------------------------------------------------------

class PassengerInput(BaseModel):
    Pclass: int = Field(..., ge=1, le=3, description="Ticket class: 1, 2, or 3")
    Sex: str = Field(..., description="'male' or 'female'")
    Age: Optional[float] = Field(None, ge=0, le=110)
    SibSp: int = Field(..., ge=0)
    Parch: int = Field(..., ge=0)
    Fare: float = Field(..., ge=0)
    Embarked: Optional[str] = Field(None, description="'S', 'C', or 'Q'")

    model_config = {
        "json_schema_extra": {
            "example": {
                "Pclass": 1, "Sex": "female", "Age": 29, "SibSp": 0,
                "Parch": 0, "Fare": 100.0, "Embarked": "S",
            }
        }
    }


class PredictionOutput(BaseModel):
    survived: bool
    survival_probability: float


# ---------------------------------------------------------------------------
# Load the pipeline ONCE at startup, via a lifespan context manager
# ---------------------------------------------------------------------------

model_state = {"pipeline": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model file not found: {MODEL_PATH}. Run Day 33's training "
            "script first to produce it."
        )
    model_state["pipeline"] = joblib.load(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")
    yield
    model_state["pipeline"] = None
    print("Model unloaded, shutting down.")


app = FastAPI(
    title="Titanic Survival Predictor",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model_state["pipeline"] is not None}


@app.post("/predict", response_model=PredictionOutput)
def predict(passenger: PassengerInput):
    pipeline = model_state["pipeline"]
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    row = pd.DataFrame([passenger.model_dump()])
    # Pydantic's Optional[...] = None fields deserialize a JSON `null` into
    # Python's None, not np.nan. Day 33 already showed SimpleImputer's default
    # missing_values=np.nan check does not recognize plain None on object
    # columns — the exact same gotcha resurfaces here, now arising naturally
    # from real API requests instead of hand-built test data. .where() below
    # converts every None cell to an actual np.nan before the pipeline runs.
    row = row.where(row.notna(), np.nan)
    prediction = int(pipeline.predict(row)[0])
    probability = float(pipeline.predict_proba(row)[:, 1][0])

    return PredictionOutput(
        survived=bool(prediction),
        survival_probability=round(probability, 4),
    )


# ---------------------------------------------------------------------------
# Verification — exercises the app in-process, no real server required
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from fastapi.testclient import TestClient

    with TestClient(app) as client:  # "with" triggers the lifespan startup/shutdown
        print("=" * 70)
        print("GET /health")
        print("=" * 70)
        resp = client.get("/health")
        print(resp.status_code, resp.json())

        test_passengers = [
            {"Pclass": 1, "Sex": "female", "Age": 29, "SibSp": 0,
             "Parch": 0, "Fare": 100.0, "Embarked": "S"},
            {"Pclass": 3, "Sex": "male", "Age": 22, "SibSp": 1,
             "Parch": 0, "Fare": 7.25, "Embarked": "S"},
            {"Pclass": 2, "Sex": "female", "Age": None, "SibSp": 1,
             "Parch": 2, "Fare": 26.0, "Embarked": None},
        ]

        print()
        print("=" * 70)
        print("POST /predict")
        print("=" * 70)
        for p in test_passengers:
            resp = client.post("/predict", json=p)
            print(p)
            print(" ->", resp.status_code, resp.json())
            print()

        print("=" * 70)
        print("POST /predict with invalid input (Pclass=5, out of the 1-3 range)")
        print("=" * 70)
        bad = {"Pclass": 5, "Sex": "female", "Age": 29, "SibSp": 0,
               "Parch": 0, "Fare": 100.0, "Embarked": "S"}
        resp = client.post("/predict", json=bad)
        print(resp.status_code, resp.json())

    print()
    print("Day 34 complete. This same app object, served for real with:")
    print("  uvicorn day34_fastapi_deployment:app --reload")
    print("answers live POST requests to http://127.0.0.1:8000/predict")
