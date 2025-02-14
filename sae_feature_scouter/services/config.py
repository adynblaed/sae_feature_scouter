#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configuration module for the SAE Feature Scouter.

Defines global defaults, model configurations, and supported visualization modes.
All parameters can be overridden via environment variables.
"""

from typing import Dict, Any
import os

DEFAULT_SEQ_LEN: int = int(os.getenv("DEFAULT_SEQ_LEN", "128"))
DEFAULT_BATCH_SIZE: int = int(os.getenv("DEFAULT_BATCH_SIZE", "256"))
DEFAULT_DEVICE: str = os.getenv("DEFAULT_DEVICE", "cuda")  # or "cpu"

MODEL_CONFIG: Dict[str, Dict[str, Any]] = {
    "basic": {
        "model_name": os.getenv("BASIC_MODEL_NAME", "gelu-1l"),
        "hook_name": os.getenv("BASIC_HOOK_NAME", "blocks.0.mlp.hook_post"),
        "dataset_path": os.getenv("BASIC_DATASET_PATH", "NeelNanda/c4-code-20k"),
    },
    "othello": {
        "model_name": os.getenv("OTHELLO_MODEL_NAME", "othello-gpt"),
        "hf_repo_id": os.getenv("OTHELLO_HF_REPO_ID", "callummcdougall/arena-demos-othellogpt"),
        "sae_id": os.getenv("OTHELLO_SAE_ID", "blocks.5.mlp.hook_post-v1"),
    },
    "attn": {
        "model_name": os.getenv("ATTN_MODEL_NAME", "attn-only-2l-demo"),
        "hf_repo_id": os.getenv("ATTN_HF_REPO_ID", "callummcdougall/arena-demos-attn2l"),
        "sae_id": os.getenv("ATTN_SAE_ID", "blocks.0.attn.hook_z-v2"),
    },
}

VIS_MODES = ["feature-centric", "prompt-centric"]
