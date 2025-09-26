# tests/conftest.py
import sys, pathlib
# Add the project root (parent of this tests/ dir) to sys.path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
