# SAE Feature Scouter

**Version:** 1.0.0  
**Build Date:** 2025-02-13

A scalable, production‐ready SAE feature scouter toolkit designed for mechanistic interpretability research. This platform provides robust visualization workflows, customizable data preparation, and a unified API interface—all configurable via environment variables. It is built with industry best practices for logging, metrics (Prometheus), and containerization.

## Features

- **Visualization Workflows:**  
  Generate dashboards for:
  - Basic Feature-Centric Visualization
  - Custom Layout Visualization
  - Prompt-Centric Visualization
  - OthelloGPT Visualization (with linear probes)
  - Attention-Only Model Visualization

- **Data Preparation:**  
  Built-in routines for tokenizing and processing data from various models.

- **Production-Grade API:**  
  Uses FastAPI with asynchronous lifespan management, structured logging via Rich, and a Prometheus `/metrics` endpoint for monitoring.

- **Configuration:**  
  All hard-coded parameters (host, port, logging, CORS, etc.) are externalized to a `.env` file for full customizability.

- **Docker-Ready:**  
  Designed for containerized deployment with a minimal Dockerfile.
Project Structure

sae_feature_scouter/
├── api/
│   └── server.py         # Consolidated FastAPI application & CLI launcher.
├── services/
│   ├── config.py         # Global configuration (env variables, defaults, model configs).
│   ├── data_loader.py    # Functions for loading and tokenizing datasets.
│   ├── data_preparation.py   # Data preparation routines for different models.
│   ├── inference.py      # Inference functions and "alive" feature extraction.
│   ├── model_loader.py   # Functions to load various transformer models and SAEs.
│   ├── storage.py        # Utilities for saving/loading generated HTML outputs.
│   ├── visualization.py  # Functions to build visualization configurations and generate dashboards.
│   └── workflow.py       # Orchestrates model loading, inference, and visualization.
├── tests/
│   └── test_api_services.py  # End-to-end API test script with reporting.
├── .env                  # Environment file with runtime parameters.
├── pyproject.toml        # Poetry project file.
└── Dockerfile            # Docker build file for containerized deployment.

Usage
Local Development

    Install Dependencies:

    Make sure you have Poetry installed, then run:

poetry install

Create a .env File:

Place a .env file in your project root. (See the Example .env section below.)

Run the API Server:

You can launch the API server with live reload enabled (for development):

poetry run python sae_feature_scouter/api/server.py --reload

CLI Options:

    --host: Server host (default: value of API_HOST in .env or 0.0.0.0).
    --port: Server port (default: value of API_PORT in .env or 8000).
    --reload: Enable live reload (useful in development).
    --workers: Number of worker processes (default: value of API_WORKERS in .env or 1).
    --log-level: Logging level (default: value of API_LOG_LEVEL in .env or info).
    --env-file: Path to your environment file (default: .env).

For example:

    poetry run python sae_feature_scouter/api/server.py --host 127.0.0.1 --port 8080 --reload --workers 2 --log-level debug

    API Endpoints:
        GET /: Returns a welcome message and API version.
        GET /health: Basic health check.
        GET /version: API version and build date.
        GET /metrics: Prometheus metrics (e.g., request count and latency).
        POST /generate: Trigger a visualization workflow. (Send a JSON payload.)
        POST /prepare/othello and POST /prepare/attn: Prepare data for specific models.

Testing the API

Run the end-to-end test script to verify all functionality and generate a report:

poetry run python sae_feature_scouter/tests/test_api_services.py \
  --device cuda \
  --tests 1,2,3,4,5 \
  --clean \
  --output-dir dashboards \
  --report-file test_report.json

This command will:

    Use the GPU (or CPU if not available).
    Run tests for demo types 1–5.
    Remove pre-existing dashboard files (if --clean is provided).
    Save dashboard HTML files to the dashboards folder.
    Output a detailed JSON report to test_report.json.

Docker Deployment

Create a Dockerfile in your project root (see below), then build and run your Docker container:

Dockerfile:

FROM python:3.10-slim

WORKDIR /app

# Copy project files.
COPY . /app

# Install Poetry and project dependencies.
RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev

# Expose the API port.
EXPOSE 8000

# Set environment variables if desired, or mount a .env file.
CMD ["poetry", "run", "python", "sae_feature_scouter/api/server.py"]

Build and Run:

docker build -t sae_feature_scouter .
docker run -p 8000:8000 --env-file .env sae_feature_scouter

Example .env

Place this file at the root of your project:

# Server Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=false
API_WORKERS=1
API_LOG_LEVEL=info

# Application Info
API_VERSION=1.0.0
BUILD_DATE=2025-02-13
API_OUTPUT_DIR=dashboards

# CORS Settings (comma-separated list of allowed origins)
CORS_ALLOW_ORIGINS=*

# Device Preference (cuda or cpu)
DEFAULT_DEVICE=cuda