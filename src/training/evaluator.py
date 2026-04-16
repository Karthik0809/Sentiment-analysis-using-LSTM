"""
Model evaluation: accuracy, F1, ROC-AUC, confusion matrix,
classification report, and per-example predictions.
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

LABEL_NAMES = ["Negative", "Neutral", "Positive"]


class Evaluator:
    """
    Evaluate any model that returns (logits, attn_weights) from forward().

    Example
    -------
    >>> ev = Evaluator(model, device)
    >>> results = ev.evaluate(test_loader)
    >>> print(results["accuracy"], results["f1_macro"])
    """

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device

    # ------------------------------------------------------------------
    # Batch evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict:
        self.model.eval()
        all_preds:  List[int]   = []
        all_labels: List[int]   = []
        all_probs:  List[List]  = []

        for sequences, labels in loader:
            sequences = sequences.to(self.device)
            logits, _ = self.model(sequences)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
            all_probs.extend(probs.tolist())

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)

        unique = np.unique(y_true)
        target_names = [LABEL_NAMES[i] for i in unique]

        metrics: Dict = {
            "accuracy":      float(accuracy_score(y_true, y_pred)),
            "f1_macro":      float(f1_score(y_true, y_pred, average="macro",    zero_division=0)),
            "f1_weighted":   float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "precision":     float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall":        float(recall_score(y_true, y_pred, average="macro",    zero_division=0)),
            "confusion_matrix":       confusion_matrix(y_true, y_pred).tolist(),
            "classification_report":  classification_report(
                y_true, y_pred, target_names=target_names, zero_division=0
            ),
            "predictions":   y_pred.tolist(),
            "labels":        y_true.tolist(),
            "probabilities": y_prob.tolist(),
        }

        # ROC-AUC (only meaningful with >1 class present)
        if len(unique) > 1:
            try:
                metrics["roc_auc"] = float(
                    roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
                )
            except ValueError:
                metrics["roc_auc"] = None
        else:
            metrics["roc_auc"] = None

        logger.info(
            f"Evaluation → Acc={metrics['accuracy']:.4f} | "
            f"F1={metrics['f1_macro']:.4f} | "
            f"ROC-AUC={metrics.get('roc_auc', 'N/A')}"
        )
        return metrics

    # ------------------------------------------------------------------
    # Single-sample prediction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self, sequence: List[int]
    ) -> Tuple[str, float, Dict[str, float], Optional[np.ndarray]]:
        """
        Returns
        -------
        sentiment   : "Positive" | "Neutral" | "Negative"
        confidence  : float  (max softmax probability)
        scores      : dict   {label: probability}
        attn        : np.ndarray | None  (attention weights if available)
        """
        self.model.eval()
        tensor = torch.tensor([sequence], dtype=torch.long).to(self.device)
        logits, attn_weights = self.model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        pred_idx = int(probs.argmax())

        attn = (
            attn_weights.squeeze().cpu().numpy()
            if attn_weights is not None else None
        )
        scores = {LABEL_NAMES[i]: float(probs[i]) for i in range(len(probs))}
        return LABEL_NAMES[pred_idx], float(probs[pred_idx]), scores, attn
