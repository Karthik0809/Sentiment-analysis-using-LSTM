"""
Root-level entry-point required by Streamlit Community Cloud.
Downloads model checkpoint from Google Drive on first run, then
delegates to app/streamlit_app.py.

Deploy:
  1. Upload checkpoints/best_model.pt to Google Drive (make it public)
  2. Paste the file ID into GDRIVE_FILE_ID below
  3. Push repo to GitHub — preprocessor.pkl is already committed
  4. Go to share.streamlit.io → New app → point to this file
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

# ── Download model weights from Google Drive if not present ───────────
# Replace this with your actual Google Drive file ID:
#   Share the file → "Anyone with the link" → copy the ID from the URL
#   https://drive.google.com/file/d/<FILE_ID>/view
GDRIVE_FILE_ID = "YOUR_GDRIVE_FILE_ID_HERE"
MODEL_PATH     = os.path.join(os.path.dirname(__file__), "checkpoints", "best_model.pt")

if not os.path.exists(MODEL_PATH) and GDRIVE_FILE_ID != "YOUR_GDRIVE_FILE_ID_HERE":
    try:
        import gdown
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        print(f"Downloading model weights from Google Drive …")
        gdown.download(id=GDRIVE_FILE_ID, output=MODEL_PATH, quiet=False)
        print("Model downloaded successfully.")
    except Exception as e:
        print(f"Warning: could not download model — {e}")

# ── Run the app ────────────────────────────────────────────────────────
exec(open(os.path.join(os.path.dirname(__file__), "app", "streamlit_app.py")).read())
