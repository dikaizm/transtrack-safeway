from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.routes import detect_router
from app.core.database import create_tables, upgrade_schema
from app.schemas.response import SuccessResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_tables()
    upgrade_schema()
    yield


app = FastAPI(
    title="Transtrack Road Safety Detection API",
    description="Async video-based road safety hazard detection for mining operations.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": exc.detail},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = "; ".join(
        f"{' -> '.join(str(l) for l in e['loc'])}: {e['msg']}" for e in exc.errors()
    )
    return JSONResponse(
        status_code=422,
        content={"status": "error", "message": errors},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": "Internal server error."},
    )


app.include_router(detect_router)


@app.get("/health", response_model=SuccessResponse)
def health():
    return SuccessResponse(message="Service is healthy.")
