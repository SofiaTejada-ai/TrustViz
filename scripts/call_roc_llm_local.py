# scripts/call_roc_llm_local.py
import sys, pathlib, os, json
from dotenv import load_dotenv, find_dotenv

# Ensure project root is on sys.path (run from anywhere)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# Load .env explicitly (even though app.py also does this)
load_dotenv(find_dotenv())

# Quick proof the script sees the key
print("SCRIPT sees key:", bool(os.environ.get("OPENAI_API_KEY")))

from fastapi.testclient import TestClient
from trustviz.server.app import app

client = TestClient(app)

payload = {
    "y_true":  [0,0,1,1,1,0,1,0,1,0],
    "y_score": [0.1,0.3,0.9,0.8,0.7,0.2,0.65,0.4,0.55,0.05],
}

r = client.post("/roc/llm", json=payload)
print("status:", r.status_code)
print(json.dumps(r.json(), indent=2)[:1200])
