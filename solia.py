"""
Entry point for launching the Solia Streamlit application.

This allows running the app with: streamlit run solia
"""

import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import and run the frontend app
from frontend.app import *
