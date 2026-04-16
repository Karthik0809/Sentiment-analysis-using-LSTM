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

# Ensure src/ is importable when running directly
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.utils.config import get_device, load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Global state (populated during startup) ───────────────────────────
MODELS: Dict[str, torch.nn.Module] = {}
TOKENIZER_OR_PREPROCESSOR = None
MODEL_TYPE: str = None   # "distilbert" | "bilstm"
DEVICE: torch.device = None
VERSION = "1.0.0"

LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}

DISTILBERT_DIR = "checkpoints/distilbert"
BILSTM_CKPT    = "checkpoints/best_model.pt"
BILSTM_PREP    = "checkpoints/preprocessor.pkl"


def _load_artifacts() -> None:
    global TOKENIZER_OR_PREPROCESSOR, DEVICE, MODEL_TYPE
    config = load_config("configs/config.yaml")
    DEVICE = get_device()

    # Try DistilBERT first (primary deployed model)
    if os.path.isdir(DISTILBERT_DIR):
        from transformers import DistilBertTokenizerFast
        from src.models.transformer import DistilBertSentiment

        model = DistilBertSentiment.from_pretrained(DISTILBERT_DIR)
        model.eval()
        MODELS["distilbert"] = model.to(DEVICE)
        TOKENIZER_OR_PREPROCESSOR = DistilBertTokenizerFast.from_pretrained(DISTILBERT_DIR)
        MODEL_TYPE = "distilbert"
        logger.info("Model loaded: distilbert (80.4% accuracy)")
        return

    # Fall back to BiLSTM
    if os.path.exists(BILSTM_CKPT) and os.path.exists(BILSTM_PREP):
        from src.models.lstm import BiLSTMWithAttention

        with open(BILSTM_PREP, "rb") as fh:
            TOKENIZER_OR_PREPROCESSOR = pickle.load(fh)

        model = BiLSTMWithAttention(
            vocab_size=TOKENIZER_OR_PREPROCESSOR.actual_vocab_size,
            embedding_dim=config["model"]["embedding_dim"],
            hidden_dim=config["model"]["hidden_dim"],
            num_classes=config["model"]["num_classes"],
            n_layers=config["model"]["n_layers"],
            dropout=0.0,
        )
        ckpt = torch.load(BILSTM_CKPT, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        MODELS["bilstm_attention"] = model.to(DEVICE)
        MODEL_TYPE = "bilstm"
        logger.info("Model loaded: bilstm_attention (76.0% accuracy)")
        return

    raise FileNotFoundError(
        f"No model checkpoint found. "
        f"Expected {DISTILBERT_DIR}/ or {BILSTM_CKPT}. "
        "Train first: python scripts/train_bert.py"
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once on startup; release on shutdown."""
    try:
        _load_artifacts()
    except FileNotFoundError as exc:
        logger.warning(
            f"Checkpoint not found ({exc}). "
            "Train the model first: python scripts/train_bert.py"
        )
    yield
    MODELS.clear()
    logger.info("Shutdown: models released.")


# ── Application ───────────────────────────────────────────────────────
app = FastAPI(
    title="News Sentiment Analysis API",
    description=(
        "REST API for multi-class news headline sentiment analysis. "
        "Powered by DistilBERT fine-tuned (80.4% accuracy, 0.919 ROC-AUC) "
        "with BiLSTM + Self-Attention fallback (76.0%)."
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

def _active_model_name() -> str:
    return "distilbert" if MODEL_TYPE == "distilbert" else "bilstm_attention"


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


def _predict_one(text: str) -> PredictionResponse:
    model_name = _active_model_name()
    model = _require_model(model_name)

    with torch.no_grad():
        if MODEL_TYPE == "distilbert":
            enc = TOKENIZER_OR_PREPROCESSOR(
                text, truncation=True, padding="max_length",
                max_length=64, return_tensors="pt",
            )
            logits, _ = model(
                enc["input_ids"].to(DEVICE),
                enc["attention_mask"].to(DEVICE),
            )
        else:
            sequence = TOKENIZER_OR_PREPROCESSOR.encode(text)
            tensor = torch.tensor([sequence], dtype=torch.long).to(DEVICE)
            logits, _ = model(tensor)

        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    pred = int(probs.argmax())
    return PredictionResponse(
        text=text,
        sentiment=LABEL_MAP[pred],
        confidence=float(probs[pred]),
        scores={LABEL_MAP[i]: float(probs[i]) for i in range(3)},
        model_used=model_name,
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
    return _predict_one(request.text)


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
    summary="Classify multiple headlines at once",
)
async def predict_batch(request: BatchPredictionRequest):
    """
    Classify up to 100 news headlines in a single request.
    """
    model_name = _active_model_name()
    model = _require_model(model_name)

    with torch.no_grad():
        if MODEL_TYPE == "distilbert":
            enc = TOKENIZER_OR_PREPROCESSOR(
                request.texts, truncation=True, padding="max_length",
                max_length=64, return_tensors="pt",
            )
            logits, _ = model(
                enc["input_ids"].to(DEVICE),
                enc["attention_mask"].to(DEVICE),
            )
        else:
            sequences = [TOKENIZER_OR_PREPROCESSOR.encode(t) for t in request.texts]
            tensor = torch.tensor(sequences, dtype=torch.long).to(DEVICE)
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
