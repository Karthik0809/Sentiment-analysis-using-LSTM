"""
Root-level entry-point required by Streamlit Community Cloud.
Delegates to app/streamlit_app.py.

Deploy:
  1. Push repo to GitHub (include checkpoints/ with best_model.pt + preprocessor.pkl)
  2. Go to share.streamlit.io → New app → point to this file
  3. Your public URL: https://<your-app>.streamlit.app
"""
import os
import sys

# Make sure src/ is importable from the project root
sys.path.insert(0, os.path.dirname(__file__))

# Re-execute the actual app module
exec(open(os.path.join(os.path.dirname(__file__), "app", "streamlit_app.py")).read())
