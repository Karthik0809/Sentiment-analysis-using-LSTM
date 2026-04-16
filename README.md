# News Headline Sentiment Analysis

An end-to-end NLP system for **three-class sentiment classification** of news headlines.
Fine-tunes DistilBERT on 204 K HuffPost articles and compares it against a custom
BiLSTM + Self-Attention baseline, achieving **80.4% test accuracy** and **0.919 ROC-AUC**.

---

**Live App: https://sentiment-newsanalysis.streamlit.app/**

1. Type or paste any news headline
2. Click **Analyse Sentiment** — get Negative / Neutral / Positive instantly
3. Switch to **Live News Feed** to fetch and analyse real headlines from BBC, CNN, NPR and more

---

## Overview

| | |
|---|---|
| Task | 3-class sentiment — Negative / Neutral / Positive |
| Dataset | HuffPost News Category Dataset — 209 K articles, 42 categories |
| Deployed model | DistilBERT fine-tuned — **80.4% accuracy, 0.919 ROC-AUC** |
| Comparison model | BiLSTM + Self-Attention — 76.0% accuracy |
| Frontend | Streamlit — Single prediction, Batch analysis, Live News Feed |
| Backend API | FastAPI — single and batch inference endpoints |
| Deployment | Streamlit Community Cloud + Docker |
| Live news | Real-time RSS from BBC, NPR, CNN, TechCrunch, The Guardian, Al Jazeera |
| Temporal trends | Hourly/daily sentiment score charts and source heatmaps |

---

## Results

### Model Comparison

| Model | Accuracy | F1 Macro | ROC-AUC | Params |
|---|---|---|---|---|
| BiLSTM + Self-Attention | 76.0% | 0.705 | 0.882 | 12.4 M |
| **DistilBERT fine-tuned** | **80.4%** | **0.753** | **0.919** | **66 M** |

DistilBERT is the deployed model. BiLSTM results are included for comparison — both models were trained on the same dataset and label scheme.

### DistilBERT Per-Class Breakdown

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Negative | 0.81 | 0.78 | 0.80 | 12,211 |
| Neutral | 0.68 | 0.53 | 0.60 | 7,129 |
| Positive | 0.83 | 0.91 | 0.87 | 21,591 |

> The Neutral class (F1 0.60) is hardest — categories like Business and Science
> contain headlines with mixed sentiment that category-based labels cannot cleanly resolve.

---

## Architecture

### Deployed: DistilBERT Fine-tuned

```
Input headline (raw text, max 64 WordPiece tokens)
       │
  DistilBERT encoder — 6 Transformer layers, 768 hidden dims, 66 M params
       │
  [CLS] representation (768-dim)
       │
  Pre-classifier FC(768 → 768) → GELU → Dropout(0.2)
       │
  Classifier FC(768 → 3)
       │
  {Negative, Neutral, Positive}
```

| Hyperparameter | Value |
|---|---|
| Base model | distilbert-base-uncased |
| Optimizer | AdamW, lr = 2e-5, warmup 10% |
| Batch size | 32 |
| Epochs | 5 (best at epoch 2) |
| Max length | 64 tokens |

### Comparison: BiLSTM + Self-Attention

```
Input tokens (max 50)
       │
  Embedding 128-dim (Xavier init, optional GloVe 6B)
       │
  Bidirectional LSTM × 3 layers (256 hidden/direction, dropout 0.4)
       │
  Additive Self-Attention → context vector + interpretable weights
       │
  LayerNorm → FC(512→256) → GELU → Dropout → FC(256→3)
       │
  {Negative, Neutral, Positive}
```

---

## Project Structure

```
Sentiment-analysis-using-LSTM/
│
├── src/
│   ├── data/
│   │   ├── preprocessor.py       # Tokenize → lemmatize → encode (BiLSTM)
│   │   ├── dataset.py            # PyTorch Dataset + DataModule (BiLSTM)
│   │   └── bert_dataset.py       # HuggingFace tokenized dataset (DistilBERT)
│   ├── models/
│   │   ├── lstm.py               # BiLSTMWithAttention, BaselineLSTM, StackedBiGRU
│   │   └── transformer.py        # DistilBertSentiment wrapper
│   ├── training/
│   │   ├── trainer.py            # AdamW + LR schedule + early stopping + MLflow
│   │   └── evaluator.py          # Accuracy, F1, AUC, confusion matrix
│   └── utils/
│       ├── config.py             # YAML config loader + device selection
│       ├── news_feed.py          # RSS fetcher + batch inference (both models)
│       ├── trend_analysis.py     # Temporal sentiment trend charts
│       ├── embeddings.py         # GloVe pre-trained embedding loader
│       └── visualization.py      # Plotly: loss curves, confusion matrix
│
├── api/
│   ├── main.py                   # FastAPI endpoints
│   └── schemas.py                # Pydantic v2 request/response models
│
├── app/
│   └── streamlit_app.py          # 4-tab Streamlit dashboard
│
├── scripts/
│   ├── train_bert.py             # Fine-tune DistilBERT (primary)
│   ├── train.py                  # Train BiLSTM/GRU/LSTM (comparison)
│   ├── evaluate.py               # Evaluate checkpoint on test set
│   ├── predict.py                # CLI inference
│   ├── compare_models.py         # Train all BiLSTM variants + comparison plot
│   └── download_glove.py         # Download GloVe 6B vectors
│
├── tests/
│   ├── test_models.py
│   └── test_preprocessor.py
│
├── deployment/
│   ├── Dockerfile
│   ├── Dockerfile.streamlit
│   └── docker-compose.yml
│
├── research/
│   └── paper.tex                 # IEEE conference paper (LaTeX)
│
├── app.py                        # Streamlit Cloud entry point
├── configs/config.yaml
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
from Kaggle and place it at `data/News_Category_Dataset_v3.json`.

### 3. Train

```bash
# Fine-tune DistilBERT — recommended (80.4% accuracy, ~55 min GPU)
python scripts/train_bert.py

# Train BiLSTM comparison model (76.0% accuracy, ~25 min GPU)
python scripts/train.py

# Optional: BiLSTM with GloVe embeddings
python scripts/download_glove.py
python scripts/train.py --glove-path data/glove/glove.6B.100d.txt
```

### 4. Run the app

```bash
streamlit run app/streamlit_app.py
```

The app loads DistilBERT from `checkpoints/distilbert/` if present, otherwise falls back to the BiLSTM checkpoint.

### 5. Run the API

```bash
uvicorn api.main:app --reload --port 8000
```

---

## API Reference

### `POST /predict`

```json
{ "text": "Scientists discover Alzheimer's cure" }
```
```json
{
  "sentiment": "Positive",
  "confidence": 0.891,
  "scores": { "Negative": 0.041, "Neutral": 0.068, "Positive": 0.891 }
}
```

### `POST /predict/batch`

```json
{ "texts": ["Headline 1", "Headline 2"] }
```

### `GET /health`

```json
{ "status": "healthy", "version": "1.0.0" }
```

---

## Deployment

### Streamlit Community Cloud

The live app at **https://sentiment-newsanalysis.streamlit.app/** is deployed from this repository.
On first boot, `app.py` downloads the DistilBERT checkpoint (~236 MB) from Google Drive and caches it in `/tmp/`.

### Docker

```bash
cd deployment && docker-compose up --build
# API:  http://localhost:8000/docs
# UI:   http://localhost:8501
```

---

## Live News Feature

```python
from src.utils.news_feed import fetch_headlines, analyze_headlines
from transformers import DistilBertTokenizerFast
from src.models.transformer import DistilBertSentiment

tokenizer = DistilBertTokenizerFast.from_pretrained("checkpoints/distilbert")
model     = DistilBertSentiment.from_pretrained("checkpoints/distilbert")

headlines = fetch_headlines("BBC News", max_items=20)
headlines = analyze_headlines(headlines, model, None, device, tokenizer=tokenizer)

for h in headlines:
    print(f"{h.sentiment:8s} ({h.confidence:.0%})  {h.title}")
```

Sources: `BBC News`, `NPR`, `CNN`, `TechCrunch`, `The Guardian`, `Al Jazeera`

---

## Research Paper

An IEEE-format conference paper is in [research/paper.tex](research/paper.tex).

Covers: dataset construction, DistilBERT fine-tuning, BiLSTM architecture comparison,
per-class analysis, and the full deployment pipeline.

```bash
cd research && pdflatex paper.tex
```

---

## Author

**Karthik Mulugu** — M.S. Computer Science (AI/ML), University at Buffalo, SUNY

- GitHub: [Karthik0809](https://github.com/Karthik0809)
- LinkedIn: [linkedin.com/in/karthik0809](https://www.linkedin.com/in/karthik0809)
- Email: karthikmulugu14@gmail.com

---

MIT License
