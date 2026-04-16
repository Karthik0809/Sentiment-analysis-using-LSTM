"""
Interactive visualizations using Plotly.
All functions return plotly.graph_objects.Figure objects that can be
rendered in Streamlit, Jupyter, or exported as HTML.
"""
from typing import Dict, List

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

LABEL_COLORS = {
    "Negative": "#EF5350",
    "Neutral":  "#42A5F5",
    "Positive": "#66BB6A",
}


def plot_training_history(
    history: Dict[str, List[float]], model_name: str = "Model"
) -> go.Figure:
    """Loss and accuracy curves side by side."""
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Loss", "Accuracy"),
    )
    # Loss
    fig.add_trace(go.Scatter(x=epochs, y=history["train_loss"], name="Train",
                             line=dict(color="#1565C0", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=history["val_loss"],   name="Validation",
                             line=dict(color="#E53935", width=2, dash="dash")), row=1, col=1)
    # Accuracy
    fig.add_trace(go.Scatter(x=epochs, y=history["train_acc"], name="Train",
                             line=dict(color="#2E7D32", width=2), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=epochs, y=history["val_acc"],   name="Validation",
                             line=dict(color="#F57F17", width=2, dash="dash"), showlegend=False), row=1, col=2)

    fig.update_layout(
        title=dict(text=f"{model_name} — Training History", font_size=16),
        height=380,
        legend=dict(orientation="h", y=-0.15),
        template="plotly_white",
    )
    fig.update_yaxes(title_text="Loss",     row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", row=1, col=2, range=[0, 1])
    fig.update_xaxes(title_text="Epoch")
    return fig


def plot_confusion_matrix(
    cm: List[List[int]], labels: List[str]
) -> go.Figure:
    """Annotated confusion matrix heatmap."""
    fig = px.imshow(
        cm,
        x=labels, y=labels,
        color_continuous_scale="Blues",
        text_auto=True,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        title="Confusion Matrix",
    )
    fig.update_layout(height=420, template="plotly_white")
    return fig


def plot_model_comparison(results: Dict[str, Dict]) -> go.Figure:
    """Grouped bar chart comparing multiple models on key metrics."""
    model_names = list(results.keys())
    metric_keys = ["accuracy", "f1_macro", "roc_auc"]
    metric_labels = ["Accuracy", "F1 (Macro)", "ROC-AUC"]

    fig = go.Figure()
    for key, label in zip(metric_keys, metric_labels):
        values = [float(results[m].get(key) or 0) for m in model_names]
        fig.add_trace(go.Bar(name=label, x=model_names, y=values, text=[f"{v:.3f}" for v in values],
                             textposition="outside"))

    fig.update_layout(
        title="Model Comparison",
        barmode="group",
        yaxis=dict(title="Score", range=[0, 1.05]),
        height=420,
        template="plotly_white",
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


def plot_attention_weights(
    tokens: List[str], weights: np.ndarray
) -> go.Figure:
    """Horizontal bar chart showing per-token attention scores."""
    # Clip to token length
    n = min(len(tokens), len(weights))
    tokens  = tokens[:n]
    weights = np.array(weights[:n], dtype=float)
    weights = weights / weights.sum()  # re-normalise for display

    colors = [
        f"rgba(21,101,192,{min(float(w) * 4, 1.0):.2f})" for w in weights
    ]
    fig = go.Figure(go.Bar(
        x=weights.tolist(),
        y=tokens,
        orientation="h",
        marker_color=colors,
        text=[f"{w:.3f}" for w in weights],
        textposition="outside",
    ))
    fig.update_layout(
        title="Token Attention Weights",
        xaxis_title="Attention Score",
        yaxis=dict(autorange="reversed"),
        height=max(300, n * 30),
        template="plotly_white",
    )
    return fig


def plot_category_distribution(category_counts: Dict[str, int]) -> go.Figure:
    """Horizontal bar chart of news category counts."""
    sorted_items = sorted(category_counts.items(), key=lambda x: x[1])
    fig = go.Figure(go.Bar(
        x=[v for _, v in sorted_items],
        y=[k for k, _ in sorted_items],
        orientation="h",
        marker_color="#1565C0",
    ))
    fig.update_layout(
        title="News Category Distribution",
        xaxis_title="Count",
        height=700,
        template="plotly_white",
    )
    return fig
