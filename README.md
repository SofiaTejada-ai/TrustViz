# TrustViz 

TrustViz is a secure visualization layer for teaching AI/cybersecurity.
It **recomputes and verifies** numbers (e.g., ROC curves) before rendering
interactive charts. It blocks unsafe content and returns **accessible** visuals.

## Run
```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn trustviz.server.app:app --reload
