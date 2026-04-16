"""
Root-level entry-point required by Streamlit Community Cloud.
Downloads model checkpoint from Google Drive on first run, then
delegates to app/streamlit_app.py.
"""
import os
import sys

# ── Ensure project root is the working directory and on sys.path ──────
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Download model weights from Google Drive if not present ───────────
GDRIVE_FILE_ID = "1Gtcx25zherCBc7iIuoyYDgd5-qJDaIpG"
MODEL_PATH     = "/tmp/best_model.pt"

if not os.path.exists(MODEL_PATH):
    try:
        import gdown
        print("Downloading model weights from Google Drive …")
        gdown.download(id=GDRIVE_FILE_ID, output=MODEL_PATH, quiet=False)
        print("Download complete.")
    except Exception as e:
        print(f"Warning: could not download model — {e}")

os.environ["SENTIMENT_MODEL_PATH"] = MODEL_PATH

# ── Run the app ────────────────────────────────────────────────────────
exec(open(os.path.join(ROOT, "app", "streamlit_app.py")).read())
