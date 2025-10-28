# trustviz/server/app.py
from fastapi import FastAPI
from trustviz.server.middleware_policy import PolicyGuard
from trustviz.server.studio_routes import router as studio_router

app = FastAPI(title="TrustViz")
app.add_middleware(PolicyGuard)
app.include_router(studio_router)

@app.get("/")
def root():
    return {"ok": True, "route": "/studio"}

@app.get("/health")
def health():
    return {"ok": True}
