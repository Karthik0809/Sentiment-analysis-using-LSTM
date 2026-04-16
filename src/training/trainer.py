"""
Unified training loop with:
  - AdamW optimizer + ReduceLROnPlateau scheduler
  - Gradient clipping
  - Early stopping
  - Best-checkpoint saving
  - JSON history export
  - MLflow experiment tracking (optional — graceful fallback if not installed)
"""
import json
import logging
import os
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

# MLflow is optional — import at module level so the flag is set once
try:
    import mlflow
    import mlflow.pytorch
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self._counter = 0
        self._best = float("inf")

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self._best - self.min_delta:
            self._best = val_loss
            self._counter = 0
        else:
            self._counter += 1
        return self._counter >= self.patience


class Trainer:
    """
    Generic trainer for any nn.Module that returns (logits, _) from forward().

    Parameters
    ----------
    model       : PyTorch module
    config      : dict loaded from configs/config.yaml
    device      : torch.device
    checkpoint_dir : folder to store .pt checkpoints and history.json
    """

    def __init__(
        self,
        model: nn.Module,
        config: dict,
        device: torch.device,
        checkpoint_dir: str = "checkpoints",
        experiment_name: Optional[str] = None,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        tcfg = config.get("training", {})
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = AdamW(
            model.parameters(),
            lr=tcfg.get("learning_rate", 1e-3),
            weight_decay=tcfg.get("weight_decay", 1e-5),
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", patience=2, factor=0.5, verbose=True
        )
        self.early_stopping = EarlyStopping(patience=tcfg.get("patience", 5))

        self.history: Dict[str, list] = {
            "train_loss": [], "val_loss": [],
            "train_acc":  [], "val_acc":  [],
        }

        # ── MLflow setup ─────────────────────────────────────────────
        self._mlflow_run = None
        if _MLFLOW_AVAILABLE:
            exp_name = experiment_name or "news-sentiment-analysis"
            mlflow.set_experiment(exp_name)
            self._mlflow_run = mlflow.start_run()
            # Log all training hyperparameters at startup
            mlflow.log_params({
                "learning_rate":  tcfg.get("learning_rate", 1e-3),
                "weight_decay":   tcfg.get("weight_decay", 1e-5),
                "batch_size":     config.get("training", {}).get("batch_size", 64),
                "epochs":         config.get("training", {}).get("epochs", 25),
                "patience":       tcfg.get("patience", 5),
                "embedding_dim":  config.get("model", {}).get("embedding_dim", 128),
                "hidden_dim":     config.get("model", {}).get("hidden_dim", 256),
                "n_layers":       config.get("model", {}).get("n_layers", 3),
                "dropout":        config.get("model", {}).get("dropout", 0.4),
                "num_classes":    config.get("model", {}).get("num_classes", 3),
            })
            logger.info(
                f"MLflow run started: {self._mlflow_run.info.run_id} "
                f"(experiment: {exp_name})"
            )
        else:
            logger.info("MLflow not installed — experiment tracking disabled. "
                        "Install with: pip install mlflow")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _one_epoch(self, loader: DataLoader, training: bool) -> tuple:
        self.model.train(training)
        total_loss = total_correct = total_samples = 0

        grad_ctx = torch.enable_grad() if training else torch.no_grad()
        with grad_ctx:
            for sequences, labels in loader:
                sequences = sequences.to(self.device)
                labels    = labels.to(self.device)

                logits, _ = self.model(sequences)
                loss = self.criterion(logits, labels)

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                total_loss    += loss.item() * labels.size(0)
                total_correct += (logits.argmax(1) == labels).sum().item()
                total_samples += labels.size(0)

        return total_loss / total_samples, total_correct / total_samples

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        epochs:       int,
    ) -> Dict[str, list]:
        best_val_acc = 0.0

        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc = self._one_epoch(train_loader, training=True)
            vl_loss, vl_acc = self._one_epoch(val_loader,   training=False)

            self.scheduler.step(vl_loss)
            self.history["train_loss"].append(tr_loss)
            self.history["val_loss"].append(vl_loss)
            self.history["train_acc"].append(tr_acc)
            self.history["val_acc"].append(vl_acc)

            lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train  loss={tr_loss:.4f}  acc={tr_acc:.4f} | "
                f"Val    loss={vl_loss:.4f}  acc={vl_acc:.4f} | "
                f"LR={lr:.2e}"
            )

            # ── Log to MLflow ─────────────────────────────────────────
            if _MLFLOW_AVAILABLE and self._mlflow_run:
                mlflow.log_metrics(
                    {
                        "train_loss": tr_loss,
                        "train_acc":  tr_acc,
                        "val_loss":   vl_loss,
                        "val_acc":    vl_acc,
                        "lr":         lr,
                    },
                    step=epoch,
                )

            if vl_acc > best_val_acc:
                best_val_acc = vl_acc
                self.save("best_model.pt")
                logger.info(f"  ✓ New best val acc: {best_val_acc:.4f}")

            if self.early_stopping(vl_loss):
                logger.info(f"Early stopping triggered at epoch {epoch}.")
                break

        self._save_history()

        # ── Finalise MLflow run ───────────────────────────────────────
        if _MLFLOW_AVAILABLE and self._mlflow_run:
            mlflow.log_metric("best_val_acc", best_val_acc)
            ckpt_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            if os.path.exists(ckpt_path):
                mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")
            mlflow.end_run()
            logger.info("MLflow run finished.")

        return self.history

    def save(self, filename: str) -> None:
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(
            {
                "model_state_dict":     self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "history":              self.history,
            },
            path,
        )

    def load(self, filename: str) -> None:
        path = os.path.join(self.checkpoint_dir, filename)
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if "history" in ckpt:
            self.history = ckpt["history"]

    def _save_history(self) -> None:
        path = os.path.join(self.checkpoint_dir, "training_history.json")
        with open(path, "w") as fh:
            json.dump(self.history, fh, indent=2)
        logger.info(f"Training history saved to {path}")
