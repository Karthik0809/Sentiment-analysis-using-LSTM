# News Headline Sentiment Analysis — BiLSTM + Self-Attention

A production-ready, end-to-end NLP system for **three-class sentiment classification**
of news headlines. Trained on 170 K+ HuffPost articles, achieving **76.3% accuracy**
and **0.743 macro-F1**.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-orange.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://your-app.streamlit.app)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![CI](https://github.com/Karthik0809/news-sentiment-bilstm/actions/workflows/ci.yml/badge.svg)](https://github.com/Karthik0809/news-sentiment-bilstm/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Live Demo & Links

| Resource | URL |
|---|---|
| Streamlit App | https://your-app.streamlit.app |
| GitHub | https://github.com/Karthik0809/news-sentiment-bilstm |
| API Docs | http://localhost:8000/docs (after running locally) |

---

## Overview

| Feature | Detail |
|---|---|
| Task | 3-class sentiment (Negative / Neutral / Positive) |
| Dataset | HuffPost News Category Dataset — 209 K articles, 42 categories |
| Best model | BiLSTM + Additive Self-Attention + LayerNorm |
| Test accuracy | **76.3%** |
| Macro F1 | **0.743** |
| REST API | FastAPI — single + batch inference |
| Frontend | Streamlit — 4 tabs including Live News Feed |
| Deployment | Docker + Streamlit Community Cloud |
| Experiment tracking | MLflow — params, metrics, artifacts |
| Inference optimization | ONNX export — 2–4× CPU speedup |
| CI/CD | GitHub Actions — test → build → publish |
| Live news | Real-time RSS from BBC, NPR, CNN, TechCrunch, The Guardian, Al Jazeera |
| Pre-trained embeddings | Optional GloVe 6B (100-dim) for improved low-frequency token coverage |
| Temporal trend analysis | Hourly/daily sentiment score trends and source heatmaps |

---

## Architecture

```
Input tokens (headline, max 50 tokens)
       │
  Embedding  128-dim  ── Xavier init (or GloVe 6B fine-tuned)
       │
  Bidirectional LSTM × 3 layers
  (256 hidden units/direction = 512 total  ·  orthogonal init  ·  dropout 0.4)
       │
  Additive Self-Attention  ──►  per-token weights (interpretable)
       │
  LayerNorm
       │
  FC(512 → 256) → GELU → Dropout(0.4)
       │
  FC(256 → 3) → Softmax
       │
  {Negative, Neutral, Positive}
```

**Key design decisions**

| Choice | Benefit |
|---|---|
| Bidirectionality | Captures both left and right context in 8–12 word headlines |
| Additive attention | Shows *which tokens* drove the prediction (interpretable) |
| LayerNorm before head | Stabilises gradient flow, +0.6% accuracy |
| GELU activation | Outperforms ReLU by ~0.3% on this dataset |
| Orthogonal LSTM init | Faster convergence, reduces vanishing gradient |
| AdamW + weight decay | Regularisation without hurting sparse gradients |
| GloVe embeddings (opt.) | Pre-trained 6B token vectors; fine-tuned during training |

---

## Results

### Model Comparison

| Model | Accuracy | F1 Macro | Params |
|---|---|---|---|
| **BiLSTM-SA (ours)** | **76.3%** | **0.743** | **12.4 M** |

> Labels derived from VADER linguistic sentiment (compound score thresholds), not news categories.
> Lower accuracy vs. category-based training reflects genuine sentiment ambiguity in news headlines.

---

## Project Structure

```
news-sentiment-bilstm/
│
├── src/                          # Core library (pip install -e .)
│   ├── data/
│   │   ├── preprocessor.py       # Clean → tokenize → lemmatize → encode
│   │   └── dataset.py            # PyTorch Dataset + stratified DataModule
│   ├── models/
│   │   └── lstm.py               # BiLSTMWithAttention, BaselineLSTM, StackedBiGRU
│   ├── training/
│   │   ├── trainer.py            # AdamW + LR schedule + early stopping + MLflow
│   │   └── evaluator.py          # Acc, F1, AUC, confusion matrix, per-example preds
│   └── utils/
│       ├── config.py             # YAML loader + device selection (CUDA/MPS/CPU)
│       ├── metrics.py            # Sklearn metric wrappers
│       ├── visualization.py      # Plotly: loss curves, confusion matrix, attention
│       ├── news_feed.py          # RSS fetcher + batch sentiment annotation
│       ├── embeddings.py         # GloVe / FastText pre-trained embedding loader
│       └── trend_analysis.py     # Temporal sentiment trend aggregation + Plotly charts
│
├── api/                          # FastAPI REST backend
│   ├── main.py                   # Lifespan model loading, CORS, endpoints
│   └── schemas.py                # Pydantic v2 request/response models
│
├── app/
│   └── streamlit_app.py          # 4-tab dashboard (Single · Batch · Live News · Arch)
│
├── scripts/
│   ├── train.py                  # CLI trainer with all hyperparameter overrides
│   ├── evaluate.py               # Checkpoint evaluation + JSON report
│   ├── predict.py                # CLI inference: single / file / JSON output
│   ├── compare_models.py         # Trains all architectures and generates comparison
│   ├── export_onnx.py            # ONNX export + verify + benchmark
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
├── .github/workflows/ci.yml      # Test → Docker build → publish on tag
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
git clone https://github.com/Karthik0809/news-sentiment-bilstm.git
cd news-sentiment-bilstm
pip install -e .
```

### 2. Get the dataset

Download [News Category Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset)
from Kaggle and place it:

```
data/News_Category_Dataset_v3.json
```

### 3. Train

```bash
python scripts/train.py                      # BiLSTM+Attention (default, ~25 min GPU)
python scripts/train.py --model bigru        # Stacked BiGRU
python scripts/train.py --model lstm         # Baseline LSTM
python scripts/compare_models.py             # All three → comparison plot
```

#### Optional: Train with GloVe pre-trained embeddings

```bash
# Download GloVe 6B vectors (~128 MB, 100-dim)
python scripts/download_glove.py

# Train using GloVe (fine-tuned during training)
python scripts/train.py --glove-path data/glove/glove.6B.100d.txt
```

### 4. Run the app

```bash
streamlit run app/streamlit_app.py           # http://localhost:8501
```

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

Output:
```
  😊  Positive (93.2%)
  Headline: Scientists discover breakthrough cancer treatment
  Scores  : Negative: 2.1% | Neutral: 4.7% | Positive: 93.2%
```

### MLflow Experiment Tracking

```bash
mlflow ui                    # http://localhost:5000
# All runs logged automatically during training
```

### ONNX Export

```bash
python scripts/export_onnx.py               # export to checkpoints/model.onnx
python scripts/export_onnx.py --verify      # confirm output matches PyTorch
python scripts/export_onnx.py --benchmark   # throughput comparison
```

### GloVe Pre-trained Embeddings

Download and use Stanford GloVe 6B vectors to improve coverage of rare tokens:

```bash
# Download 100-dim vectors to data/glove/ (~128 MB)
python scripts/download_glove.py

# Download all four dimensions (50/100/200/300-dim)
python scripts/download_glove.py --all

# Train with GloVe — embedding layer is fine-tuned
python scripts/train.py --glove-path data/glove/glove.6B.100d.txt
python scripts/train.py --model bigru --glove-path data/glove/glove.6B.100d.txt
```

### Temporal Trend Analysis

The Live News Feed tab automatically shows sentiment trends over time after fetching
headlines. You can also use the analysis functions directly:

```python
from src.utils.trend_analysis import compute_trend, sentiment_score_series, make_trend_figure

# Group analyzed headlines by publication hour and compute stats
df = compute_trend(headlines, bucket="hour")          # or bucket="day"
print(df[["time_bucket", "mean_score", "positive_pct", "count"]])

# Per-source trend (returns {"All Sources": df, "BBC News": df, ...})
series = sentiment_score_series(headlines, bucket="hour")
fig    = make_trend_figure(series)                    # Plotly line chart
fig.show()
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
  "confidence": 0.912,
  "scores": { "Negative": 0.031, "Neutral": 0.057, "Positive": 0.912 },
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
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select `app.py` as the entry point
4. Your live URL: `https://<your-app>.streamlit.app`

> `app.py` automatically downloads `best_model.pt` (~112 MB) from the
> GitHub Release asset on first boot — no manual file upload needed.

### Option 2 — Docker (local / cloud server)

```bash
cd deployment
docker-compose up --build
# API:  http://localhost:8000/docs
# UI:   http://localhost:8501
```

### Option 3 — Docker Hub + Cloud (AWS/GCP/Azure)

Tag a release (`git tag v1.0.0 && git push --tags`) to trigger GitHub Actions automatic push to Docker Hub, then pull and run on any cloud VM.

---

## Live News Feature

Fetch and analyse real headlines from major sources programmatically:

```python
from src.utils.news_feed import fetch_headlines, analyze_headlines
from src.utils.trend_analysis import compute_trend, make_trend_figure

headlines = fetch_headlines("BBC News", max_items=20)
headlines = analyze_headlines(headlines, model, preprocessor, device)

for h in headlines:
    print(f"{h.sentiment:8s} ({h.confidence:.0%})  {h.title}")

# Temporal trend — group by hour and plot sentiment score over time
df  = compute_trend(headlines, bucket="hour")
fig = make_trend_figure({"BBC News": df})
fig.show()
```

Available sources: `BBC News`, `NPR`, `CNN`, `TechCrunch`, `The Guardian`, `Al Jazeera`

The Streamlit dashboard renders these trends automatically in the Live News tab:
- **Sentiment trend line chart** — mean score (−1 to +1) per hour or day, one line per source
- **Source × Sentiment heatmap** — % breakdown of Negative / Neutral / Positive per source

---

## Research Paper

A complete 8-section IEEE-format conference paper is in [research/paper.tex](research/paper.tex).

Covers: motivation, related work (LSTM/BERT survey), dataset construction,
BiLSTM-SA architecture with full mathematical formulation, ablation study,
DistilBERT comparison, and the full deployment pipeline.

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
