"""
Temporal sentiment trend analysis using RSS feed data.

Groups analyzed headlines by publication time and computes sentiment
statistics per time bucket (hourly or daily), enabling visualization
of how news sentiment shifts over a time window.
"""
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

LABEL_NAMES  = ["Negative", "Neutral", "Positive"]
LABEL_SCORES = {"Negative": -1.0, "Neutral": 0.0, "Positive": 1.0}


def compute_trend(
    headlines,                        # List[NewsHeadline] with .sentiment set
    bucket: str = "hour",             # "hour" | "day"
    source_filter: Optional[str] = None,
) -> pd.DataFrame:
    """
    Aggregate per-headline sentiment scores into time buckets.

    Parameters
    ----------
    headlines     : analyzed NewsHeadline objects (must have .published + .sentiment)
    bucket        : "hour" groups by publication hour; "day" groups by date
    source_filter : if set, only include headlines from this source

    Returns
    -------
    DataFrame with columns:
        time_bucket  – datetime label for the bucket
        mean_score   – average sentiment score (-1 negative … +1 positive)
        positive_pct – fraction of positive headlines
        neutral_pct  – fraction of neutral headlines
        negative_pct – fraction of negative headlines
        count        – number of headlines in the bucket
    """
    records = []
    for h in headlines:
        if source_filter and h.source != source_filter:
            continue
        if not h.sentiment or not h.published:
            continue
        try:
            dt = datetime.strptime(h.published, "%Y-%m-%d %H:%M")
        except ValueError:
            continue

        records.append({
            "time":      dt,
            "sentiment": h.sentiment,
            "score":     LABEL_SCORES.get(h.sentiment, 0.0),
            "source":    h.source,
            "headline":  h.title,
        })

    if not records:
        return pd.DataFrame(columns=["time_bucket", "mean_score",
                                     "positive_pct", "neutral_pct",
                                     "negative_pct", "count"])

    df = pd.DataFrame(records)

    # Truncate timestamps to the chosen bucket
    if bucket == "hour":
        df["time_bucket"] = df["time"].apply(
            lambda t: t.replace(minute=0, second=0, microsecond=0)
        )
    else:
        df["time_bucket"] = df["time"].apply(
            lambda t: t.replace(hour=0, minute=0, second=0, microsecond=0)
        )

    grouped = (
        df.groupby("time_bucket")
        .apply(lambda g: pd.Series({
            "mean_score":   g["score"].mean(),
            "positive_pct": (g["sentiment"] == "Positive").mean(),
            "neutral_pct":  (g["sentiment"] == "Neutral").mean(),
            "negative_pct": (g["sentiment"] == "Negative").mean(),
            "count":        len(g),
        }))
        .reset_index()
        .sort_values("time_bucket")
    )
    return grouped


def sentiment_score_series(
    headlines,
    bucket: str = "hour",
) -> Dict[str, pd.DataFrame]:
    """
    Compute trend per source (and one combined 'All Sources' series).

    Returns a dict of {source_name: DataFrame from compute_trend()}.
    """
    sources = list({h.source for h in headlines if h.source})
    result = {"All Sources": compute_trend(headlines, bucket=bucket)}
    for src in sources:
        df = compute_trend(headlines, bucket=bucket, source_filter=src)
        if not df.empty:
            result[src] = df
    return result


def make_trend_figure(trend_data: Dict[str, pd.DataFrame], bucket: str = "hour"):
    """
    Build a Plotly figure with one line per source showing sentiment over time.
    Returns a go.Figure ready for st.plotly_chart().
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    colours = [
        "#1565C0", "#E53935", "#2E7D32", "#F57F17",
        "#6A1B9A", "#00838F", "#D84315",
    ]

    for i, (src, df) in enumerate(trend_data.items()):
        if df.empty:
            continue
        color = colours[i % len(colours)]
        fig.add_trace(go.Scatter(
            x=df["time_bucket"].tolist(),
            y=df["mean_score"].tolist(),
            mode="lines+markers",
            name=src,
            line=dict(color=color, width=2),
            marker=dict(size=6),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Sentiment score: %{y:.2f}<br>"
                f"Source: {src}<extra></extra>"
            ),
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.5,
                  annotation_text="Neutral", annotation_position="right")
    fig.update_layout(
        title=f"Sentiment Trend by {bucket.title()}",
        xaxis_title="Publication Time",
        yaxis_title="Mean Sentiment Score  (−1 = Negative, +1 = Positive)",
        yaxis=dict(range=[-1.1, 1.1]),
        height=420,
        template="plotly_white",
        legend=dict(orientation="h", y=-0.2),
        hovermode="x unified",
    )
    return fig


def make_heatmap_figure(headlines, sources: List[str]):
    """
    Source × Sentiment count heatmap using hourly bins.
    Rows = news source, Columns = Negative / Neutral / Positive.
    """
    import plotly.graph_objects as go
    import numpy as np

    matrix = []
    for src in sources:
        row = []
        src_h = [h for h in headlines if h.source == src and h.sentiment]
        total = max(len(src_h), 1)
        for label in LABEL_NAMES:
            row.append(sum(1 for h in src_h if h.sentiment == label) / total * 100)
        matrix.append(row)

    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=LABEL_NAMES,
        y=sources,
        colorscale="RdYlGn",
        zmin=0, zmax=100,
        text=[[f"{v:.0f}%" for v in row] for row in matrix],
        texttemplate="%{text}",
        hovertemplate="Source: %{y}<br>%{x}: %{z:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title="Sentiment Distribution by Source (%)",
        height=max(300, len(sources) * 50 + 120),
        template="plotly_white",
    )
    return fig
