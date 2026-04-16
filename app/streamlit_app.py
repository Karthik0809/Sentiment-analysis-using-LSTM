"""
Streamlit demo application for the News Sentiment Analyzer.

Run:
    streamlit run app/streamlit_app.py
"""
import os
import sys

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="News Sentiment Analyzer",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

LABEL_COLORS = {"Negative": "#EF5350", "Neutral": "#42A5F5", "Positive": "#66BB6A"}
LABEL_EMOJI  = {"Negative": "😠", "Neutral": "😐", "Positive": "😊"}
LABEL_NAMES  = ["Negative", "Neutral", "Positive"]

SAMPLE_HEADLINES = [
    "Scientists discover breakthrough in Alzheimer's treatment",
    "Stock market plunges amid recession fears",
    "Local sports team wins national championship",
    "Government corruption scandal rocks capital",
    "New restaurant opens serving plant-based cuisine downtown",
    "Climate change threatens coastal cities by 2050",
    "Couple celebrates 70th wedding anniversary with community party",
    "Unemployment rate rises to highest level in decade",
]


# ── Model loading (cached) ────────────────────────────────────────────
# Google Drive IDs — update DISTILBERT_GDRIVE_ID after uploading the zipped model
BILSTM_GDRIVE_ID    = "1VqAN-wLViwMJMjQYOxeMEQfIEJx72Ihl"
DISTILBERT_GDRIVE_ID = ""   # set after training & uploading checkpoints/distilbert.tar.gz

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ensure_distilbert() -> str:
    """
    Return path to the DistilBERT checkpoint directory.
    Downloads and extracts from Google Drive if not present locally.
    Returns "" if DISTILBERT_GDRIVE_ID is not set.
    """
    local = os.path.join(ROOT_DIR, "checkpoints", "distilbert")
    tmp   = "/tmp/distilbert"

    if os.path.isdir(local) and os.path.exists(os.path.join(local, "config.json")):
        return local
    if os.path.isdir(tmp) and os.path.exists(os.path.join(tmp, "config.json")):
        return tmp
    if not DISTILBERT_GDRIVE_ID:
        return ""

    import gdown, tarfile
    archive = "/tmp/distilbert.tar.gz"
    st.info("Downloading DistilBERT weights (~260 MB) — this happens once …")
    gdown.download(id=DISTILBERT_GDRIVE_ID, output=archive, quiet=False)
    with tarfile.open(archive) as tar:
        tar.extractall("/tmp")
    return tmp


def _ensure_bilstm() -> str:
    """Return path to BiLSTM best_model.pt, downloading from Google Drive if needed."""
    local = os.path.join(ROOT_DIR, "checkpoints", "best_model.pt")
    tmp   = "/tmp/best_model.pt"
    if os.path.exists(local):
        return local
    if os.path.exists(tmp):
        return tmp
    import gdown
    st.info("Downloading BiLSTM weights (~112 MB) — this happens once …")
    gdown.download(id=BILSTM_GDRIVE_ID, output=tmp, quiet=False)
    return tmp


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    """
    Load the best available model.
    Priority: DistilBERT (higher accuracy) → BiLSTM (fallback).
    Returns (model, preprocessor_or_tokenizer, device, model_type).
    """
    import pickle
    from src.utils.config import get_device, load_config

    config = load_config("configs/config.yaml")
    device = get_device()

    # ── Try DistilBERT first ──────────────────────────────────────────
    bert_dir = _ensure_distilbert()
    if bert_dir:
        from transformers import DistilBertTokenizerFast
        from src.models.transformer import DistilBertSentiment

        tokenizer = DistilBertTokenizerFast.from_pretrained(bert_dir)
        model     = DistilBertSentiment.from_pretrained(bert_dir)
        model.eval()
        return model.to(device), tokenizer, device, "distilbert"

    # ── Fall back to BiLSTM ───────────────────────────────────────────
    from src.models.lstm import BiLSTMWithAttention

    with open(os.path.join(ROOT_DIR, "checkpoints", "preprocessor.pkl"), "rb") as fh:
        preprocessor = pickle.load(fh)

    model = BiLSTMWithAttention(
        vocab_size    = preprocessor.actual_vocab_size,
        embedding_dim = config["model"]["embedding_dim"],
        hidden_dim    = config["model"]["hidden_dim"],
        num_classes   = config["model"]["num_classes"],
        n_layers      = config["model"]["n_layers"],
        dropout       = 0.0,
    )
    ckpt = torch.load(_ensure_bilstm(), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model.to(device), preprocessor, device, "bilstm"


def predict(text: str, model, tok_or_prep, device, model_type: str):
    """Unified predict for DistilBERT or BiLSTM."""
    if model_type == "distilbert":
        enc = tok_or_prep(
            text, return_tensors="pt", truncation=True,
            padding="max_length", max_length=64,
        )
        with torch.no_grad():
            logits, _ = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        return int(probs.argmax()), probs, [], None   # no token-level attention

    # BiLSTM path
    seq    = tok_or_prep.encode(text)
    tensor = torch.tensor([seq], dtype=torch.long).to(device)
    with torch.no_grad():
        logits, attn_weights = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    pred   = int(probs.argmax())
    tokens = tok_or_prep.preprocess(text)[: tok_or_prep.max_len]
    attn   = (
        attn_weights.squeeze().cpu().numpy()[: len(tokens)]
        if attn_weights is not None else None
    )
    return pred, probs, tokens, attn


# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📰 News Sentiment")
    st.title("About")
    if model_type == "distilbert":
        st.info(
            "**Model**: DistilBERT (fine-tuned)\n\n"
            "**Dataset**: HuffPost News (~170K headlines)\n\n"
            "**Classes**: Negative / Neutral / Positive\n\n"
            "**Architecture**: 66M-param transformer, 6 layers, "
            "768 hidden dims"
        )
    else:
        st.info(
            "**Model**: BiLSTM + Self-Attention\n\n"
            "**Dataset**: HuffPost News (~170K headlines)\n\n"
            "**Classes**: Negative / Neutral / Positive\n\n"
            "**Architecture**: 3-layer BiLSTM, 256 hidden units, "
            "additive attention, LayerNorm"
        )
    st.divider()
    st.subheader("Try an Example")
    for ex in SAMPLE_HEADLINES:
        if st.button(ex[:55] + "…" if len(ex) > 55 else ex, key=ex):
            st.session_state["headline_input"] = ex


# ── Main area ─────────────────────────────────────────────────────────
st.title("📰 News Headline Sentiment Analyzer")
_model_label = "DistilBERT (fine-tuned)" if model_type == "distilbert" else "Bidirectional LSTM with Self-Attention"
st.caption(
    f"Powered by {_model_label} trained on "
    "200 K+ news articles across 42 categories."
)

# Load model (non-blocking warning if not available)
model_available = False
model_type = "bilstm"
try:
    model, tok_or_prep, device, model_type = load_model()
    model_available = True
except Exception as exc:
    st.warning(
        f"Model not loaded ({exc}).  \n"
        "Run `python scripts/train.py` or `python scripts/train_bert.py` to train a checkpoint."
    )

# ── Tab layout ────────────────────────────────────────────────────────
tab_single, tab_batch, tab_live, tab_about = st.tabs(
    ["Single Prediction", "Batch Analysis", "Live News Feed", "Model Architecture"]
)

# ──────────────────────────────────────────────────────────────────────
# Tab 1: Single Prediction
# ──────────────────────────────────────────────────────────────────────
with tab_single:
    headline = st.text_area(
        "Enter a news headline:",
        value=st.session_state.get("headline_input", ""),
        height=90,
        placeholder="e.g., Scientists discover breakthrough in cancer treatment...",
        key="headline_input",
    )

    if st.button("Analyse Sentiment", type="primary", disabled=not model_available):
        if headline.strip():
            with st.spinner("Analysing…"):
                pred_idx, probs, tokens, attn = predict(
                    headline, model, tok_or_prep, device, model_type
                )
            sentiment = LABEL_NAMES[pred_idx]
            color = LABEL_COLORS[sentiment]

            # Result banner
            st.markdown(
                f"""<div style='background:{color}22;border-left:4px solid {color};
                padding:1rem 1.5rem;border-radius:6px;margin:1rem 0'>
                <h2 style='color:{color};margin:0'>{LABEL_EMOJI[sentiment]} {sentiment}</h2>
                <p style='margin:.3rem 0 0 0;color:#555'>
                    Confidence: <strong>{probs[pred_idx]:.1%}</strong>
                </p></div>""",
                unsafe_allow_html=True,
            )

            # Score cards
            cols = st.columns(3)
            for i, name in enumerate(LABEL_NAMES):
                with cols[i]:
                    st.metric(f"{LABEL_EMOJI[name]} {name}", f"{probs[i]:.1%}")
                    st.progress(float(probs[i]))

            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(float(probs[2]) * 100, 1),
                title={"text": "Positivity Score (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#66BB6A"},
                    "steps": [
                        {"range": [0,  40], "color": "#FFCDD2"},
                        {"range": [40, 65], "color": "#E3F2FD"},
                        {"range": [65, 100], "color": "#C8E6C9"},
                    ],
                },
            ))
            fig_gauge.update_layout(height=260, margin=dict(t=40, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Attention weights
            if attn is not None and len(tokens) > 0:
                st.subheader("Token Attention Weights")
                st.caption(
                    "Tokens the model weighted most heavily when making its decision."
                )
                n = min(len(tokens), len(attn))
                tok_slice = tokens[:n]
                att_slice = attn[:n] / (attn[:n].sum() + 1e-9)

                fig_attn = go.Figure(go.Bar(
                    x=tok_slice,
                    y=att_slice.tolist(),
                    marker_color=[
                        f"rgba(21,101,192,{min(float(w)*4,1):.2f})" for w in att_slice
                    ],
                    text=[f"{w:.3f}" for w in att_slice],
                    textposition="outside",
                ))
                fig_attn.update_layout(
                    xaxis_title="Token", yaxis_title="Attention",
                    height=300, template="plotly_white",
                )
                st.plotly_chart(fig_attn, use_container_width=True)
        else:
            st.warning("Please enter a headline.")

# ──────────────────────────────────────────────────────────────────────
# Tab 2: Batch Analysis
# ──────────────────────────────────────────────────────────────────────
with tab_batch:
    st.markdown("Enter one headline per line (max 200).")
    batch_input = st.text_area(
        "Headlines:",
        height=200,
        placeholder="Headline 1\nHeadline 2\n...",
    )

    if st.button("Analyse All", disabled=not model_available) and batch_input.strip():
        lines = [h.strip() for h in batch_input.splitlines() if h.strip()][:200]
        results = []
        progress = st.progress(0)
        for i, line in enumerate(lines):
            idx, probs, _, _ = predict(line, model, tok_or_prep, device, model_type)
            results.append({
                "Headline":   line,
                "Sentiment":  LABEL_NAMES[idx],
                "Confidence": f"{probs[idx]:.1%}",
                "Positive":   f"{probs[2]:.1%}",
                "Neutral":    f"{probs[1]:.1%}",
                "Negative":   f"{probs[0]:.1%}",
            })
            progress.progress((i + 1) / len(lines))

        import pandas as pd
        df = pd.DataFrame(results)

        # Colour-coded table
        def highlight_sentiment(val):
            colours = {"Positive": "#C8E6C9", "Neutral": "#E3F2FD", "Negative": "#FFCDD2"}
            return f"background-color: {colours.get(val, '')}"

        st.dataframe(
            df.style.applymap(highlight_sentiment, subset=["Sentiment"]),
            use_container_width=True,
        )

        # Sentiment distribution pie
        counts = df["Sentiment"].value_counts()
        fig_pie = go.Figure(go.Pie(
            labels=counts.index.tolist(),
            values=counts.values.tolist(),
            marker_colors=[LABEL_COLORS[s] for s in counts.index],
            hole=0.35,
        ))
        fig_pie.update_layout(title="Sentiment Distribution", height=360)
        st.plotly_chart(fig_pie, use_container_width=True)

        # Download CSV
        st.download_button(
            "Download Results (CSV)",
            data=df.to_csv(index=False),
            file_name="sentiment_results.csv",
            mime="text/csv",
        )

# ──────────────────────────────────────────────────────────────────────
# Tab 3: Live News Feed
# ──────────────────────────────────────────────────────────────────────
with tab_live:
    st.subheader("Live News Sentiment Feed")
    st.caption(
        "Fetch real headlines from major news outlets and analyse their sentiment in real-time."
    )

    try:
        from src.utils.news_feed import RSS_FEEDS, analyze_headlines, fetch_multi_source
        from src.utils.trend_analysis import (
            make_heatmap_figure, make_trend_figure, sentiment_score_series
        )

        feed_available = True
    except ImportError:
        feed_available = False
        st.warning("Install feedparser to enable live news: `pip install feedparser`")

    if feed_available:
        col_src, col_n = st.columns([3, 1])
        with col_src:
            selected_sources = st.multiselect(
                "News sources",
                options=list(RSS_FEEDS.keys()),
                default=["BBC News"],
            )
        with col_n:
            max_per = st.number_input(
                "Headlines per source", min_value=5, max_value=50, value=15
            )

        col_fetch, col_auto = st.columns([1, 3])
        with col_fetch:
            fetch_btn = st.button(
                "Fetch & Analyse", type="primary", disabled=not model_available
            )
        with col_auto:
            st.caption("Tip: Select multiple sources for a broader sentiment overview.")

        if fetch_btn and selected_sources:
            with st.spinner(f"Fetching from {', '.join(selected_sources)}…"):
                headlines = fetch_multi_source(selected_sources, max_per_source=max_per)

            if not headlines:
                st.error("Could not fetch headlines. Check your internet connection.")
            else:
                with st.spinner(f"Analysing {len(headlines)} headlines…"):
                    bert_tok = tok_or_prep if model_type == "distilbert" else None
                    headlines = analyze_headlines(
                        headlines, model, tok_or_prep, device, tokenizer=bert_tok
                    )

                import pandas as pd

                df = pd.DataFrame([
                    {
                        "Source":     h.source,
                        "Headline":   h.title,
                        "Sentiment":  h.sentiment,
                        "Confidence": f"{h.confidence:.1%}",
                        "Positive":   f"{h.scores.get('Positive', 0):.1%}",
                        "Neutral":    f"{h.scores.get('Neutral', 0):.1%}",
                        "Negative":   f"{h.scores.get('Negative', 0):.1%}",
                        "Published":  h.published,
                    }
                    for h in headlines
                ])

                # ── Summary KPIs ──────────────────────────────────────
                counts = df["Sentiment"].value_counts()
                total  = len(df)
                kcols  = st.columns(4)
                kcols[0].metric("Total Headlines", total)
                kcols[1].metric(
                    f"{LABEL_EMOJI['Positive']} Positive",
                    f"{counts.get('Positive', 0)}",
                    delta=f"{counts.get('Positive', 0)/total:.0%}",
                )
                kcols[2].metric(
                    f"{LABEL_EMOJI['Neutral']} Neutral",
                    f"{counts.get('Neutral', 0)}",
                    delta=f"{counts.get('Neutral', 0)/total:.0%}",
                )
                kcols[3].metric(
                    f"{LABEL_EMOJI['Negative']} Negative",
                    f"{counts.get('Negative', 0)}",
                    delta=f"-{counts.get('Negative', 0)/total:.0%}",
                    delta_color="inverse",
                )

                # ── Charts row ───────────────────────────────────────
                ch1, ch2 = st.columns(2)
                with ch1:
                    fig_pie = go.Figure(go.Pie(
                        labels=counts.index.tolist(),
                        values=counts.values.tolist(),
                        marker_colors=[LABEL_COLORS[s] for s in counts.index],
                        hole=0.4,
                        textinfo="label+percent",
                    ))
                    fig_pie.update_layout(
                        title="Overall Sentiment Split", height=320,
                        showlegend=False,
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                with ch2:
                    per_source = (
                        df.groupby(["Source", "Sentiment"])
                        .size()
                        .reset_index(name="Count")
                    )
                    fig_bar = go.Figure()
                    for sent, color in LABEL_COLORS.items():
                        sub = per_source[per_source["Sentiment"] == sent]
                        fig_bar.add_trace(go.Bar(
                            name=sent,
                            x=sub["Source"].tolist(),
                            y=sub["Count"].tolist(),
                            marker_color=color,
                        ))
                    fig_bar.update_layout(
                        title="Sentiment by Source", barmode="stack",
                        height=320, legend=dict(orientation="h", y=-0.2),
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                # ── Colour-coded table ────────────────────────────────
                def _bg(val):
                    c = {"Positive": "#C8E6C9", "Neutral": "#E3F2FD", "Negative": "#FFCDD2"}
                    return f"background-color: {c.get(val, '')}"

                st.dataframe(
                    df.style.applymap(_bg, subset=["Sentiment"]),
                    use_container_width=True,
                    height=400,
                )

                # ── Temporal Trend Analysis ──────────────────────────
                st.divider()
                st.subheader("Sentiment Trend Over Time")
                st.caption(
                    "Groups headlines by their publication timestamp to reveal "
                    "how sentiment shifts across the fetched window."
                )

                bucket_choice = st.radio(
                    "Group by", ["hour", "day"], horizontal=True, key="bucket"
                )
                trend_series = sentiment_score_series(headlines, bucket=bucket_choice)

                if any(not v.empty for v in trend_series.values()):
                    fig_trend = make_trend_figure(trend_series, bucket=bucket_choice)
                    st.plotly_chart(fig_trend, use_container_width=True)

                    # Source × Sentiment heatmap
                    st.subheader("Source Sentiment Heatmap")
                    src_list = [s for s in selected_sources
                                if any(h.source == s and h.sentiment for h in headlines)]
                    if src_list:
                        fig_heat = make_heatmap_figure(headlines, src_list)
                        st.plotly_chart(fig_heat, use_container_width=True)
                else:
                    st.info(
                        "Trend chart requires publication timestamps in the RSS feed. "
                        "Some sources omit them — try BBC News or AP News."
                    )

                # ── Download ─────────────────────────────────────────
                st.download_button(
                    "Download CSV",
                    data=df.to_csv(index=False),
                    file_name=f"live_news_sentiment_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                )


# ──────────────────────────────────────────────────────────────────────
# Tab 4: Model Architecture
# ──────────────────────────────────────────────────────────────────────
with tab_about:
    active_label = "DistilBERT" if model_type == "distilbert" else "BiLSTM + Self-Attention"
    st.subheader(f"Active Model: {active_label}")
    st.markdown("""
    ```
    Input (headline tokens)
         │
    ┌────▼────────────────────┐
    │  Embedding Layer         │  vocab_size × 128 dims
    │  (Xavier initialisation) │
    └────┬────────────────────┘
         │
    ┌────▼────────────────────────────────────────┐
    │  Bidirectional LSTM  (3 layers, 256 hidden) │
    │  Orthogonal weight init, dropout=0.4        │
    └────┬────────────────────────────────────────┘
         │  [B, T, 512]
    ┌────▼──────────────────────┐
    │  Additive Self-Attention   │  → context vector [B, 512]
    │  + attention weights       │     + interpretable weights
    └────┬──────────────────────┘
         │
    ┌────▼─────────────────┐
    │  LayerNorm [B, 512]   │
    └────┬─────────────────┘
         │
    ┌────▼──────────────────────────┐
    │  FC(512→256) → GELU → Drop    │
    └────┬──────────────────────────┘
         │
    ┌────▼─────────────────┐
    │  FC(256→3) → Logits   │
    └──────────────────────┘
    ```

    **Training details**
    | Hyperparameter | Value |
    |---|---|
    | Optimiser | AdamW |
    | Learning rate | 1e-3 (ReduceLROnPlateau ×0.5) |
    | Weight decay | 1e-5 |
    | Batch size | 64 |
    | Max epochs | 25 |
    | Early stopping | patience = 5 |
    | Grad clip norm | 1.0 |
    | Loss | CrossEntropyLoss |
    """)

    st.subheader("Dataset")
    st.markdown("""
    - **Source**: HuffPost News Category Dataset (Kaggle)
    - **Size**: ~209 K articles (2012–2022)
    - **Classes**: Negative (0) / Neutral (1) / Positive (2)
    - **Label mapping**: category → sentiment (12 positive, 8 negative, 9 neutral categories)
    """)
