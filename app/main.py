import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from app.api.v1 import stream, anpr
from app.core.logging_config import logger
from app.services.plate_service import start_loading, models_status
import os
import uvicorn

BASE_DIR = Path(__file__).resolve().parent.parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Port opens immediately; models load in a background thread.
    start_loading()
    yield


app = FastAPI(title="PiTech Platform", lifespan=lifespan)


@app.get("/health")
async def health():
    """Returns ready=true once all models are loaded."""
    return models_status()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Logs every request (method, path, status, duration) to logs/app.log
    for basic system monitoring — request volume, slow endpoints, errors."""
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(f"Unhandled error on {request.method} {request.url.path}")
        raise
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        f"{request.method} {request.url.path} -> {response.status_code} "
        f"({duration_ms:.1f}ms)"
    )
    return response


app.mount("/static", StaticFiles(directory=BASE_DIR / "frontend"), name="static")

app.include_router(anpr.router)
app.include_router(stream.router) 
# Mount static files
if not os.path.exists("static"):
    os.makedirs("static")
    
# Create output directory
os.makedirs("outputImgs", exist_ok=True)

logger.info("ANPR platform starting up...")

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    return (BASE_DIR / "frontend/dashboard.html").read_text(encoding="utf-8")

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=4000, reload=True)