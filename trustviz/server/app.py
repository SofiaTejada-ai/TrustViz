# trustviz/server/app.py
from fastapi import FastAPI
from trustviz.server.middleware_policy import PolicyGuard
from trustviz.server.studio_routes import router as studio_router
from trustviz.server import edu_routes

# trustviz/server/app.py — FastAPI app factory

from fastapi import FastAPI
from trustviz.server.studio_routes import router as studio_router
from trustviz.server.rules_routes import router as rules_router

app = FastAPI(title="ScholarViz")

# Mount routers
app.include_router(studio_router)
app.include_router(rules_router)
app.add_middleware(PolicyGuard)
app.include_router(studio_router)
app.include_router(edu_routes.router)
@app.get("/")
def root():
    return {"ok": True, "route": "/studio"}

@app.get("/health")
def health():
    return {"ok": True}
