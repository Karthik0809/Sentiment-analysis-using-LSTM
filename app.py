"""
Root-level entry-point required by Streamlit Community Cloud.
Model download logic lives in app/streamlit_app.py load_model().
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

exec(open(os.path.join(ROOT, "app", "streamlit_app.py")).read())
