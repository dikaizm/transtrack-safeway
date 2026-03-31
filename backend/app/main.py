from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.api.routes import detect_router
from app.core.database import create_tables


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_tables()
    yield


app = FastAPI(
    title="Transtrack Road Safety Detection API",
    description="Async video-based road safety hazard detection for mining operations.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(detect_router)


@app.get("/health")
def health():
    return {"status": "ok"}
