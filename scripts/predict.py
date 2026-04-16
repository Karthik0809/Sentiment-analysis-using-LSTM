#!/usr/bin/env python3
"""
CLI inference script — predict sentiment for one or more headlines.

Usage
-----
# Single headline
python scripts/predict.py --text "Scientists cure rare childhood disease"

# From a text file (one headline per line)
python scripts/predict.py --file headlines.txt

# Output as JSON
python scripts/predict.py --text "Market hits record high" --json
"""
import argparse
import json
import logging
import os
import pickle
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.models.lstm import BiLSTMWithAttention
from src.utils.config import get_device, load_config

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("predict")

LABEL_NAMES = ["Negative", "Neutral", "Positive"]
LABEL_EMOJI  = {"Negative": "😠", "Neutral": "😐", "Positive": "😊"}


def load_model():
    config = load_config("configs/config.yaml")
    device = get_device()

    with open("checkpoints/preprocessor.pkl", "rb") as fh:
        preprocessor = pickle.load(fh)

    model = BiLSTMWithAttention(
        vocab_size=preprocessor.actual_vocab_size,
        embedding_dim=config["model"]["embedding_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_classes=config["model"]["num_classes"],
        n_layers=config["model"]["n_layers"],
        dropout=0.0,
    )
    ckpt = torch.load("checkpoints/best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model.to(device), preprocessor, device


def predict_texts(texts, model, preprocessor, device):
    results = []
    sequences = [preprocessor.encode(t) for t in texts]
    tensor = torch.tensor(sequences, dtype=torch.long).to(device)
    with torch.no_grad():
        logits, _ = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    for i, text in enumerate(texts):
        pred = int(probs[i].argmax())
        results.append({
            "text":       text,
            "sentiment":  LABEL_NAMES[pred],
            "confidence": round(float(probs[i][pred]), 4),
            "scores": {LABEL_NAMES[j]: round(float(probs[i][j]), 4) for j in range(3)},
        })
    return results


def parse_args():
    p = argparse.ArgumentParser(description="Predict sentiment of news headlines")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--text", type=str, help="Single headline string")
    g.add_argument("--file", type=str, help="Path to file with one headline per line")
    p.add_argument("--json", action="store_true", help="Output JSON instead of plain text")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        model, preprocessor, device = load_model()
    except FileNotFoundError:
        logger.error("Model checkpoint not found. Run: python scripts/train.py")
        sys.exit(1)

    if args.text:
        texts = [args.text]
    else:
        with open(args.file, "r", encoding="utf-8") as fh:
            texts = [line.strip() for line in fh if line.strip()]

    results = predict_texts(texts, model, preprocessor, device)

    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        for r in results:
            emoji = LABEL_EMOJI[r["sentiment"]]
            print(f"\n  {emoji}  {r['sentiment']} ({r['confidence']:.1%})")
            print(f"  Headline: {r['text']}")
            print(
                "  Scores  : "
                + " | ".join(f"{k}: {v:.1%}" for k, v in r["scores"].items())
            )


if __name__ == "__main__":
    main()
