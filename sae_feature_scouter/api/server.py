#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SAE Feature Scouter API

This FastAPI application exposes endpoints to trigger visualization workflows,
prepare model-specific data, and provide health, version, and Prometheus metrics endpoints.
It is fully configurable via environment variables (loaded from a .env file) and is designed
for production-grade, scalable, containerized deployment.
"""

import os
import time
import uvicorn
import argparse
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

from fastapi import FastAPI, Request, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Prometheus client for metrics endpoint.
from prometheus_client import (
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
)

# Use Rich for enhanced logging.
from rich.logging import RichHandler

# Load environment variables from .env if present.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Application configuration from environment variables.
HOST = os.getenv("API_HOST", "0.0.0.0")
PORT = int(os.getenv("API_PORT", "8000"))
RELOAD = os.getenv("API_RELOAD", "false").lower() == "true"
WORKERS = int(os.getenv("API_WORKERS", "1"))
LOG_LEVEL = os.getenv("API_LOG_LEVEL", "info")
API_VERSION = os.getenv("API_VERSION", "1.0.0")
BUILD_DATE = os.getenv("BUILD_DATE", "2025-02-13")
OUTPUT_DIR = os.getenv("API_OUTPUT_DIR", "dashboards")
CORS_ALLOW_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")

# Configure logging.
logging.basicConfig(
    level=LOG_LEVEL.upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()],
)
logger = logging.getLogger("sae_feature_scouter_api")

# Create a dedicated Prometheus registry to avoid duplicate metrics on reload.
PROM_REGISTRY = CollectorRegistry(auto_describe=True)
REQUEST_COUNT = Counter(
    "sae_feature_scouter_requests_total",
    "Total number of requests",
    ["method", "endpoint", "http_status"],
    registry=PROM_REGISTRY,
)
REQUEST_LATENCY = Histogram(
    "sae_feature_scouter_request_duration_seconds",
    "Request latency in seconds",
    ["endpoint"],
    registry=PROM_REGISTRY,
)

# ---------------------------
# Lifespan Handler
# ---------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("SAE Feature Scouter API is starting up...")
    # (Initialize shared resources here, e.g., DB connections, ML models, etc.)
    yield
    logger.info("SAE Feature Scouter API is shutting down...")
    # (Clean up resources here)

# ---------------------------
# FastAPI App Initialization
# ---------------------------
app = FastAPI(lifespan=lifespan, title="SAE Feature Scouter API", version=API_VERSION)

# Enable CORS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Logging & Metrics Middleware
# ---------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    try:
        request_body = await request.body()
        payload_length = len(request_body) if request_body else 0
    except Exception:
        payload_length = 0

    idempotency_key = request.headers.get("X-Idempotency-Key")
    if idempotency_key:
        logger.info("Idempotency key provided: %s", idempotency_key)

    logger.info("Incoming request: %s %s | Payload size: %d bytes", request.method, request.url.path, payload_length)
    response = await call_next(request)
    process_time = time.time() - start_time

    # Record Prometheus metrics.
    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path, http_status=response.status_code).inc()
    REQUEST_LATENCY.labels(endpoint=request.url.path).observe(process_time)

    logger.info("Completed %s %s in %.2fms with status %d", request.method, request.url.path, process_time * 1000, response.status_code)
    return response

# ---------------------------
# API Endpoints
# ---------------------------
@app.get("/")
async def root() -> Dict[str, Any]:
    """Returns a welcome message and API version."""
    return {"message": "Welcome to the SAE Feature Scouter API", "version": app.version}

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Simple health check endpoint."""
    return {"status": "ok", "timestamp": time.time()}

@app.get("/version")
async def version_info() -> Dict[str, Any]:
    """Returns API version, build date, and description."""
    return {"version": app.version, "build": BUILD_DATE, "description": "Production-ready SAE Feature Scouter API"}

@app.get("/metrics")
async def metrics() -> Response:
    """Prometheus metrics endpoint."""
    return Response(generate_latest(PROM_REGISTRY), media_type=CONTENT_TYPE_LATEST)

# ---------------------------
# Demo Request Model
# ---------------------------
class DemoRequest(BaseModel):
    """
    Request model for triggering a visualization workflow.

    Attributes:
      - demo_type (int): Identifier for the demo workflow (1-5).
      - output_filename (str): Desired output HTML filename.
      - prompt (Optional[str]): Prompt text for prompt-centric visualization.
      - feature (int): SAE feature index.
      - token_limit (int): Maximum tokens to process.
      - metric (str): Visualization metric.
    """
    demo_type: int
    output_filename: str = "output.html"
    prompt: Optional[str] = None
    feature: int = 8
    token_limit: int = 8192
    metric: str = "act_quantile"

@app.post("/generate", response_model=Dict[str, str])
async def generate_demo(request: DemoRequest) -> Dict[str, str]:
    """
    Generate a visualization dashboard based on demo type and parameters.
    """
    try:
        from sae_feature_scouter.services.workflow import run_workflow
        logger.info("Received demo generation request: %s", request.dict())
        output_filename = run_workflow(
            demo_type=request.demo_type,
            output_filename=request.output_filename,
            prompt=request.prompt,
            feature=request.feature,
            token_limit=request.token_limit,
            metric=request.metric,
        )
        return {"status": "success", "filename": output_filename}
    except Exception as e:
        logger.error("Demo generation failed: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/prepare/othello", response_model=Dict[str, Any])
async def prepare_othello(device: str = Query(default="cuda" if os.getenv("CUDA_VISIBLE_DEVICES") or os.name != "nt" else "cpu", description="Computation device (cuda or cpu)")) -> Dict[str, Any]:
    """
    Prepare data for the OthelloGPT model.
    """
    try:
        from sae_feature_scouter.services.data_preparation import prepare_othello_data
        logger.info("Preparing OthelloGPT data on device: %s", device)
        data = prepare_othello_data(device)
        return {"status": "success", "data_keys": ["tokens", "target_logits", "linear_probes", "alive_feats"]}
    except Exception as e:
        logger.error("Failed to prepare OthelloGPT data: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/prepare/attn", response_model=Dict[str, Any])
async def prepare_attn(device: str = Query(default="cuda" if os.getenv("CUDA_VISIBLE_DEVICES") or os.name != "nt" else "cpu", description="Computation device (cuda or cpu)")) -> Dict[str, Any]:
    """
    Prepare data for the attention-only model.
    """
    try:
        from sae_feature_scouter.services.data_preparation import prepare_attn_data
        logger.info("Preparing attention-only model data on device: %s", device)
        data = prepare_attn_data(device)
        return {"status": "success", "data_keys": ["tokens", "alive_feats"]}
    except Exception as e:
        logger.error("Failed to prepare attention-only model data: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------
# CLI Launch Options
# ---------------------------
def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for launching the API server.
    """
    parser = argparse.ArgumentParser(description="Launch the SAE Feature Scouter API server with enhanced options.")
    parser.add_argument("--host", type=str, default=HOST, help=f"Server host (default: {HOST})")
    parser.add_argument("--port", type=int, default=PORT, help=f"Server port (default: {PORT})")
    parser.add_argument("--reload", action="store_true", default=RELOAD, help="Enable live reload (development only)")
    parser.add_argument("--workers", type=int, default=WORKERS, help=f"Number of worker processes (default: {WORKERS})")
    parser.add_argument("--log-level", type=str, default=LOG_LEVEL, choices=["debug", "info", "warning", "error", "critical"], help=f"Logging level (default: {LOG_LEVEL})")
    parser.add_argument("--env-file", type=str, default=".env", help="Path to an environment file (default: .env)")
    return parser.parse_args()

def main() -> None:
    """
    Main function to launch the API server using uvicorn with custom options.
    """
    args = parse_args()
    if os.path.exists(args.env_file):
        try:
            from dotenv import load_dotenv
            load_dotenv(args.env_file)
            logger.info("Loaded environment variables from '%s'.", args.env_file)
        except ImportError:
            logger.warning("python-dotenv not installed; skipping .env file loading.")
    logger.info("Starting SAE Feature Scouter API on %s:%d (reload=%s, workers=%d)",
                args.host, args.port, args.reload, args.workers)
    uvicorn.run(
        "sae_feature_scouter.api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
        workers=args.workers,
    )

if __name__ == "__main__":
    main()
