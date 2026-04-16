"""
Root-level entry-point required by Streamlit Community Cloud.
Downloads model checkpoint from GitHub Releases on first run, then
delegates to app/streamlit_app.py.

Deploy:
  1. Push repo to GitHub — preprocessor.pkl is already committed
  2. Go to share.streamlit.io → New app → point to this file
  Model weights are fetched automatically from the GitHub release asset.
"""
import os
import sys
import urllib.request

# ── Ensure project root is the working directory and on sys.path ──────
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Download model weights from GitHub Releases if not present ────────
MODEL_URL  = (
    "https://github.com/Karthik0809/Sentiment-analysis-using-LSTM"
    "/releases/download/v1.0.0/best_model.pt"
)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "best_model.pt")

if not os.path.exists(MODEL_PATH):
    try:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        print("Downloading model weights from GitHub Releases …")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded successfully.")
    except Exception as e:
        print(f"Warning: could not download model — {e}")

# ── Run the app ────────────────────────────────────────────────────────
exec(open(os.path.join(os.path.dirname(__file__), "app", "streamlit_app.py")).read())
