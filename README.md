# News Headline Sentiment Analysis — DistilBERT Fine-tuning

A production-ready, end-to-end NLP system for **three-class sentiment classification**
of news headlines. Trained on 204 K+ HuffPost articles, with a fine-tuned DistilBERT
achieving **80.4% test accuracy** and **0.919 ROC-AUC**.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-orange.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://your-app.streamlit.app)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![CI](https://github.com/Karthik0809/Sentiment-analysis-using-LSTM/actions/workflows/ci.yml/badge.svg)](https://github.com/Karthik0809/Sentiment-analysis-using-LSTM/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Live Demo & Links

| Resource | URL |
|---|---|
| Streamlit App | https://your-app.streamlit.app |
| GitHub | https://github.com/Karthik0809/Sentiment-analysis-using-LSTM |
| API Docs | http://localhost:8000/docs (after running locally) |

---

## Overview

| Feature | Detail |
|---|---|
| Task | 3-class sentiment (Negative / Neutral / Positive) |
| Dataset | HuffPost News Category Dataset — 209 K articles, 42 categories |
| Primary model | DistilBERT (`distilbert-base-uncased`) fine-tuned |
| Test accuracy | **80.4%** |
| F1 Macro | **0.753** |
| ROC-AUC | **0.919** |
| Comparison model | BiLSTM + Additive Self-Attention + LayerNorm (76.0%) |
| REST API | FastAPI — single + batch inference |
| Frontend | Streamlit — 4 tabs including Live News Feed |
| Deployment | Docker + Streamlit Community Cloud |
| Experiment tracking | MLflow — params, metrics, artifacts |
| CI/CD | GitHub Actions — unit tests on Python 3.10 and 3.11 |
| Live news | Real-time RSS from BBC, NPR, CNN, TechCrunch, The Guardian, Al Jazeera |
| Pre-trained embeddings | Optional GloVe 6B (100-dim) for BiLSTM variant |
| Temporal trend analysis | Hourly/daily sentiment score trends and source heatmaps |

---

## Architecture

### Primary: DistilBERT Fine-tuned

```
Input headline (raw text, max 64 WordPiece tokens)
       │
  DistilBERT encoder (6 Transformer layers, 768 hidden dims)
  distilbert-base-uncased — 66 M parameters
       │
  [CLS] token representation  ─► 768-dim
       │
  Pre-classifier FC(768 → 768) → GELU
       │
  Dropout(0.2)
       │
  Classifier FC(768 → 3) → Logits
       │
  {Negative, Neutral, Positive}
```

**Fine-tuning details**

| Hyperparameter | Value |
|---|---|
| Base model | distilbert-base-uncased |
| Optimizer | AdamW |
| Learning rate | 2e-5 (linear warmup 10%) |
| Weight decay | 0.01 |
| Batch size | 32 |
| Epochs | 5 (best at epoch 2) |
| Max sequence length | 64 tokens |
| Gradient clip norm | 1.0 |

### Comparison: BiLSTM + Self-Attention

```
Input tokens (headline, max 50 tokens)
       │
  Embedding  128-dim  ── Xavier init (or GloVe 6B fine-tuned)
       │
  Bidirectional LSTM × 3 layers
  (256 hidden units/direction = 512 total · dropout 0.4)
       │
  Additive Self-Attention ──► per-token weights (interpretable)
       │
  LayerNorm
       │
  FC(512 → 256) → GELU → Dropout(0.4)
       │
  FC(256 → 3) → Softmax
       │
  {Negative, Neutral, Positive}
```

---

## Results

### Model Comparison

| Model | Accuracy | F1 Macro | ROC-AUC | Params |
|---|---|---|---|---|
| BiLSTM + Self-Attention | 76.0% | 0.705 | 0.882 | 12.4 M |
| **DistilBERT fine-tuned** | **80.4%** | **0.753** | **0.919** | **66 M** |

> DistilBERT achieves **+4.4% accuracy** and **+3.7% ROC-AUC** over the BiLSTM baseline,
> with stronger generalization on the Negative and Positive classes.

### Per-Class Breakdown (DistilBERT)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Negative | 0.81 | 0.78 | 0.80 | 12,211 |
| Neutral | 0.68 | 0.53 | 0.60 | 7,129 |
| Positive | 0.83 | 0.91 | 0.87 | 21,591 |
| **Overall** | **0.78** | **0.74** | **0.75** | **40,931** |

> The Neutral class is hardest (F1 0.60) because news categories like Science and Business
> do not map cleanly to a single sentiment from text alone.

---

## Project Structure

```
Sentiment-analysis-using-LSTM/
│
├── src/                          # Core library (pip install -e .)
│   ├── data/
│   │   ├── preprocessor.py       # Clean → tokenize → lemmatize → encode (BiLSTM)
│   │   ├── dataset.py            # PyTorch Dataset + stratified DataModule (BiLSTM)
│   │   └── bert_dataset.py       # HuggingFace tokenized dataset + DataModule (DistilBERT)
│   ├── models/
│   │   ├── lstm.py               # BiLSTMWithAttention, BaselineLSTM, StackedBiGRU
│   │   └── transformer.py        # DistilBertSentiment wrapper (fine-tuning)
│   ├── training/
│   │   ├── trainer.py            # AdamW + LR schedule + early stopping + MLflow
│   │   └── evaluator.py          # Acc, F1, AUC, confusion matrix, per-example preds
│   └── utils/
│       ├── config.py             # YAML loader + device selection (CUDA/MPS/CPU)
│       ├── metrics.py            # Sklearn metric wrappers
│       ├── visualization.py      # Plotly: loss curves, confusion matrix, attention
│       ├── news_feed.py          # RSS fetcher + batch sentiment annotation (both models)
│       ├── embeddings.py         # GloVe / FastText pre-trained embedding loader
│       └── trend_analysis.py     # Temporal sentiment trend aggregation + Plotly charts
│
├── api/                          # FastAPI REST backend
│   ├── main.py                   # Lifespan model loading, CORS, endpoints
│   └── schemas.py                # Pydantic v2 request/response models
│
├── app/
│   └── streamlit_app.py          # 4-tab dashboard (Single · Batch · Live News · Arch)
│                                 # Auto-selects DistilBERT if available, falls back to BiLSTM
│
├── scripts/
│   ├── train_bert.py             # Fine-tune DistilBERT (primary — recommended)
│   ├── train.py                  # Train BiLSTM/GRU/LSTM (comparison)
│   ├── evaluate.py               # Checkpoint evaluation + JSON report
│   ├── predict.py                # CLI inference: single / file / JSON output
│   ├── compare_models.py         # Trains all BiLSTM architectures + comparison plot
│   ├── export_onnx.py            # ONNX export + verify + benchmark (BiLSTM)
│   └── download_glove.py         # Download GloVe 6B word vectors from Stanford NLP
│
├── notebooks/
│   └── original_exploration.ipynb  # Original EDA and model experiments
│
├── tests/
│   ├── test_models.py            # Output shape, attention sum, NaN checks
│   └── test_preprocessor.py      # Cleaning, label mapping, encoding tests
│
├── deployment/
│   ├── Dockerfile                # Multi-stage FastAPI image
│   ├── Dockerfile.streamlit      # Streamlit image
│   └── docker-compose.yml        # One-command full stack
│
├── research/
│   └── paper.tex                 # IEEE conference paper (LaTeX)
│
├── .github/workflows/ci.yml      # Unit tests on Python 3.10 and 3.11
├── .streamlit/config.toml        # Streamlit theme
├── app.py                        # Root entry-point for Streamlit Community Cloud
├── packages.txt                  # System deps for Streamlit Cloud
├── configs/config.yaml           # All hyperparameters
├── requirements.txt
└── setup.py
```

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/Karthik0809/Sentiment-analysis-using-LSTM.git
cd Sentiment-analysis-using-LSTM
pip install -e .
```

### 2. Get the dataset

Download [News Category Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset)
from Kaggle and place it:

```
data/News_Category_Dataset_v3.json
```

### 3. Train

#### Recommended: DistilBERT (80.4% test accuracy)

```bash
python scripts/train_bert.py                         # 5 epochs, ~55 min GPU
python scripts/train_bert.py --epochs 3 --lr 2e-5    # Faster run
```

#### Alternative: BiLSTM + Self-Attention (76.0% test accuracy)

```bash
python scripts/train.py                              # BiLSTM+Attention (~25 min GPU)
python scripts/train.py --model bigru                # Stacked BiGRU
python scripts/train.py --model lstm                 # Baseline LSTM

# Optional: train with GloVe pre-trained embeddings
python scripts/download_glove.py
python scripts/train.py --glove-path data/glove/glove.6B.100d.txt
```

### 4. Run the app

```bash
streamlit run app/streamlit_app.py           # http://localhost:8501
```

The app automatically loads DistilBERT from `checkpoints/distilbert/` if present,
otherwise falls back to BiLSTM from `checkpoints/best_model.pt`.

### 5. Run the API

```bash
uvicorn api.main:app --reload --port 8000    # http://localhost:8000/docs
```

---

## All Commands

### CLI Prediction

```bash
# Single headline
python scripts/predict.py --text "Scientists discover breakthrough cancer treatment"

# From file (one headline per line)
python scripts/predict.py --file my_headlines.txt

# JSON output
python scripts/predict.py --text "Market crashes amid recession fears" --json
```

### MLflow Experiment Tracking

```bash
mlflow ui                    # http://localhost:5000
# All BiLSTM runs logged automatically during training
```

### Run Tests

```bash
pytest tests/ -v --cov=src
```

---

## API Reference

Start: `uvicorn api.main:app --port 8000`  ·  Docs: `http://localhost:8000/docs`

### `POST /predict`

```json
// Request
{ "text": "Scientists discover Alzheimer's cure" }

// Response
{
  "text": "Scientists discover Alzheimer's cure",
  "sentiment": "Positive",
  "confidence": 0.891,
  "scores": { "Negative": 0.041, "Neutral": 0.068, "Positive": 0.891 },
  "model_used": "bilstm_attention"
}
```

### `POST /predict/batch`

```json
// Request
{ "texts": ["Headline 1", "Headline 2", "..."] }
```

### `GET /health`

```json
{ "status": "healthy", "models_loaded": ["bilstm_attention"], "version": "1.0.0" }
```

---

## Deployment

### Option 1 — Streamlit Community Cloud (free public URL)

1. Push this repo to GitHub — `preprocessor.pkl` is already committed
2. Upload `checkpoints/distilbert.tar.gz` to Google Drive and set `DISTILBERT_GDRIVE_ID` in `app/streamlit_app.py`
3. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
4. Select `app.py` as the entry point
5. Your live URL: `https://<your-app>.streamlit.app`

> On first boot, `app.py` downloads `distilbert.tar.gz` (~236 MB) from Google Drive
> and extracts it to `/tmp/distilbert/`. Subsequent boots use the cached copy.

### Option 2 — Docker (local / cloud server)

```bash
cd deployment
docker-compose up --build
# API:  http://localhost:8000/docs
# UI:   http://localhost:8501
```

### Option 3 — Docker Hub + Cloud (AWS/GCP/Azure)

Tag a release (`git tag v1.0.0 && git push --tags`) to trigger GitHub Actions
automatic push to Docker Hub, then pull and run on any cloud VM.

---

## Live News Feature

Fetch and analyse real headlines from major sources programmatically:

```python
from src.utils.news_feed import fetch_headlines, analyze_headlines
from src.utils.trend_analysis import compute_trend, make_trend_figure
from transformers import DistilBertTokenizerFast
from src.models.transformer import DistilBertSentiment

tokenizer = DistilBertTokenizerFast.from_pretrained("checkpoints/distilbert")
model     = DistilBertSentiment.from_pretrained("checkpoints/distilbert")

headlines = fetch_headlines("BBC News", max_items=20)
headlines = analyze_headlines(headlines, model, None, device, tokenizer=tokenizer)

for h in headlines:
    print(f"{h.sentiment:8s} ({h.confidence:.0%})  {h.title}")

# Temporal trend — group by hour and plot sentiment score over time
df  = compute_trend(headlines, bucket="hour")
fig = make_trend_figure({"BBC News": df})
fig.show()
```

Available sources: `BBC News`, `NPR`, `CNN`, `TechCrunch`, `The Guardian`, `Al Jazeera`

---

## Research Paper

A complete IEEE-format conference paper is in [research/paper.tex](research/paper.tex).

Covers: motivation, related work, dataset construction, DistilBERT fine-tuning,
BiLSTM-SA architecture comparison, results analysis, and deployment pipeline.

Compile:
```bash
cd research && pdflatex paper.tex && bibtex paper && pdflatex paper.tex
```

---

## Author

**Karthik Mulugu**  
M.S. in Computer Science (AI/ML) — University at Buffalo, SUNY

- GitHub: [Karthik0809](https://github.com/Karthik0809)
- LinkedIn: [linkedin.com/in/karthik0809](https://www.linkedin.com/in/karthik0809)
- Email: karthikmulugu14@gmail.com

---

## License

MIT License — see [LICENSE](LICENSE) for details.
