"""
AI Assistant Swarm

A powerful, extensible AI assistant system that uses a swarm of specialized agents 
to provide high-quality responses across different domains.
"""

__version__ = "0.1.0"

__author__ = 'AI Assistant Swarm Team'

import os
from pathlib import Path

# Set up package-level constants
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = ROOT_DIR / "data"

# Create data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True) 