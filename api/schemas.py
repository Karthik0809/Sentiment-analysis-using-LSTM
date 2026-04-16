"""Pydantic request / response schemas for the prediction API."""
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="News headline or sentence to analyse",
        examples=["Scientists discover new treatment for Alzheimer's disease"],
    )
    model_name: Optional[str] = Field(
        default="bilstm_attention",
        description="Model identifier to use for inference",
    )


class BatchPredictionRequest(BaseModel):
    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of headlines (max 100)",
    )
    model_name: Optional[str] = Field(default="bilstm_attention")


class PredictionResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    scores: Dict[str, float]
    model_used: str


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    model_used: str
    total: int


class HealthResponse(BaseModel):
    status: str
    models_loaded: List[str]
    version: str


class ModelInfoResponse(BaseModel):
    name: str
    description: str
    accuracy: Optional[float]
    f1_macro: Optional[float]
    num_params: Optional[int]
