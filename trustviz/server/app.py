# trustviz/server/app.py — FastAPI app factory (clean + complete)

from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

# Middleware
from trustviz.server.middleware_policy import PolicyGuard

# Routers
from trustviz.server.studio_routes import router as studio_router
from trustviz.server.rules_routes import router as rules_router
from trustviz.server.lesson_routes import router as lesson_router
from trustviz.server import edu_routes  # exposes edu_routes.router

app = FastAPI()

# ===== Middleware =====
app.add_middleware(PolicyGuard)

# ===== Routers (each exactly once) =====
app.include_router(studio_router)          # /scholarviz, /ask, /diagram/*, notebook, docs, export, etc.
app.include_router(rules_router)           # your rules endpoints
app.include_router(lesson_router)          # /lesson/plan, /lesson/plan_from_mermaid
app.include_router(edu_routes.router)      # your edu endpoints

# ===== Static: /static → trustviz/static =====
# Resolve package path robustly (works from repo root or installed package)
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Optional health root
@app.get("/")
def root():
    return {"ok": True, "msg": "TrustViz running"}
