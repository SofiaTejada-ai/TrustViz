from fastapi import FastAPI
from .studio_routes import router as studio_router
from .lesson_routes import router as lesson_router   # ← ADD

app = FastAPI()
app.include_router(studio_router)
app.include_router(lesson_router)                    # ← ADD
