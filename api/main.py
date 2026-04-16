"""
FastAPI application entry-point.

Run locally:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Swagger UI:  http://localhost:8000/docs
ReDoc:       http://localhost:8000/redoc
"""
import logging
import os
import pickle
import sys
from contextlib import asynccontextmanager
from typing import Dict

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Ensure src/ is importable when running directly
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.models.lstm import BiLSTMWithAttention
from src.utils.config import get_device, load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Global state (populated during startup) ───────────────────────────
MODELS: Dict[str, torch.nn.Module] = {}
PREPROCESSOR = None
DEVICE: torch.device = None
VERSION = "1.0.0"

LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}


def _load_artifacts() -> None:
    global PREPROCESSOR, DEVICE
    config = load_config("configs/config.yaml")
    DEVICE = get_device()

    prep_path = config["api"]["preprocessor_path"]
    model_path = config["api"]["model_path"]

    with open(prep_path, "rb") as fh:
        PREPROCESSOR = pickle.load(fh)

    model = BiLSTMWithAttention(
        vocab_size=PREPROCESSOR.actual_vocab_size,
        embedding_dim=config["model"]["embedding_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_classes=config["model"]["num_classes"],
        n_layers=config["model"]["n_layers"],
        dropout=0.0,   # no dropout at inference
    )
    ckpt = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    MODELS["bilstm_attention"] = model.to(DEVICE)
    logger.info("Model loaded: bilstm_attention")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once on startup; release on shutdown."""
    try:
        _load_artifacts()
    except FileNotFoundError as exc:
        logger.warning(
            f"Checkpoint not found ({exc}). "
            "Train the model first: python scripts/train.py"
        )
    yield
    MODELS.clear()
    logger.info("Shutdown: models released.")


# ── Application ───────────────────────────────────────────────────────
app = FastAPI(
    title="News Sentiment Analysis API",
    description=(
        "REST API for multi-class news headline sentiment analysis "
        "powered by a Bidirectional LSTM with Self-Attention."
    ),
    version=VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────

def _require_model(name: str) -> torch.nn.Module:
    if name not in MODELS:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model '{name}' is not loaded. "
                "Train and checkpoint the model first."
            ),
        )
    return MODELS[name]


def _predict_one(text: str, model: torch.nn.Module) -> PredictionResponse:
    sequence = PREPROCESSOR.encode(text)
    tensor = torch.tensor([sequence], dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        logits, _ = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    pred = int(probs.argmax())
    return PredictionResponse(
        text=text,
        sentiment=LABEL_MAP[pred],
        confidence=float(probs[pred]),
        scores={LABEL_MAP[i]: float(probs[i]) for i in range(3)},
        model_used="bilstm_attention",
    )


# ── Endpoints ─────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "News Sentiment Analysis API", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Service liveness probe."""
    return HealthResponse(
        status="healthy",
        models_loaded=list(MODELS.keys()),
        version=VERSION,
    )


@app.get("/models", response_model=list, tags=["System"])
async def list_models():
    """List available model names."""
    return list(MODELS.keys())


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Classify a single headline",
)
async def predict(request: PredictionRequest):
    """
    Classify one news headline as **Positive**, **Neutral**, or **Negative**.

    Returns the predicted class, confidence score, and softmax scores for
    all three classes.
    """
    model = _require_model(request.model_name or "bilstm_attention")
    return _predict_one(request.text, model)


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
    summary="Classify multiple headlines at once",
)
async def predict_batch(request: BatchPredictionRequest):
    """
    Classify up to 100 news headlines in a single request.
    Inputs are batched for efficient GPU utilisation.
    """
    model_name = request.model_name or "bilstm_attention"
    model = _require_model(model_name)

    sequences = [PREPROCESSOR.encode(t) for t in request.texts]
    tensor = torch.tensor(sequences, dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        logits, _ = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    predictions = []
    for i, text in enumerate(request.texts):
        pred = int(probs[i].argmax())
        predictions.append(
            PredictionResponse(
                text=text,
                sentiment=LABEL_MAP[pred],
                confidence=float(probs[i][pred]),
                scores={LABEL_MAP[j]: float(probs[i][j]) for j in range(3)},
                model_used=model_name,
            )
        )

    return BatchPredictionResponse(
        predictions=predictions,
        model_used=model_name,
        total=len(predictions),
    )
