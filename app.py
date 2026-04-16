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
MODEL_URL = (
    "https://github.com/Karthik0809/Sentiment-analysis-using-LSTM"
    "/releases/download/v1.0.0/best_model.pt"
)

# Try repo checkpoints/ first (local dev); fall back to /tmp/ on
# Streamlit Cloud where the repo filesystem is read-only.
_REPO_PATH = os.path.join(ROOT, "checkpoints", "best_model.pt")
_TMP_PATH  = "/tmp/best_model.pt"

if os.path.exists(_REPO_PATH):
    MODEL_PATH = _REPO_PATH
elif os.path.exists(_TMP_PATH):
    MODEL_PATH = _TMP_PATH
else:
    # Attempt repo path first, fall back to /tmp/
    for _dest in (_REPO_PATH, _TMP_PATH):
        try:
            os.makedirs(os.path.dirname(_dest), exist_ok=True)
            print(f"Downloading model weights to {_dest} …")
            urllib.request.urlretrieve(MODEL_URL, _dest)
            print("Download complete.")
            MODEL_PATH = _dest
            break
        except Exception as e:
            print(f"Could not write to {_dest}: {e}")
    else:
        MODEL_PATH = _REPO_PATH  # let the app show its own error

# Expose path so streamlit_app.py can read it
os.environ["SENTIMENT_MODEL_PATH"] = MODEL_PATH

# ── Run the app ────────────────────────────────────────────────────────
exec(open(os.path.join(ROOT, "app", "streamlit_app.py")).read())
