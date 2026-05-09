"""Pytest configuration — adds project root to sys.path."""
import sys
from pathlib import Path

# Ensure project root is on sys.path so all modules resolve correctly
sys.path.insert(0, str(Path(__file__).parent))
