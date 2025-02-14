#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Workflow Orchestrator for the SAE Feature Scouter.

Composes model loading, inference, and visualization services to generate outputs
based on the specified demo type.
"""

import logging
from typing import Any, Dict
from sae_feature_scouter.services.model_loader import load_basic_model, load_othello_model, load_attn_model
from sae_feature_scouter.services.visualization import generate_feature_centric_vis, generate_prompt_centric_vis
from sae_feature_scouter.services.config import DEFAULT_SEQ_LEN

logger = logging.getLogger(__name__)

def run_workflow(demo_type: int, **kwargs: Any) -> str:
    """
    Execute a visualization workflow based on demo type and parameters.

    Args:
        demo_type (int): Demo identifier (1-5).
        **kwargs: Additional parameters (output_filename, prompt, feature, token_limit, metric).

    Returns:
        str: The output HTML filename.

    Raises:
        ValueError: If demo type is unsupported.
        Exception: For any workflow errors.
    """
    try:
        logger.info("Starting workflow for demo type %d with parameters: %s", demo_type, kwargs)
        output_filename = kwargs.get("output_filename", f"demo_{demo_type}.html")
        if demo_type in [1, 2, 3]:
            data = load_basic_model(DEFAULT_SEQ_LEN)
            model = data["model"]
            sae = data["sae"]
            tokens = data["tokens"][: kwargs.get("token_limit", 8192)]
            if demo_type == 3:
                prompt = kwargs.get("prompt", "'first_name': ('django.db.models.fields")
                try:
                    seq_tokens = model.tokenizer.tokenize(prompt)
                    seq_pos = seq_tokens.index("Ġ('")
                except Exception:
                    seq_pos = 0
                    logger.warning("Could not locate expected token in prompt; defaulting seq_pos to 0.")
                metric = kwargs.get("metric", "act_quantile")
                return generate_prompt_centric_vis(model, sae, tokens, prompt, seq_pos, metric, output_filename)
            else:
                extra_config = {}
                if demo_type == 2:
                    extra_config["layout"] = kwargs.get("layout_config")
                    extra_config["features"] = kwargs.get("features", list(range(256)))
                    tokens = tokens[:4096, :48]
                return generate_feature_centric_vis(model, sae, tokens, output_filename, feature=kwargs.get("feature", 8), extra_config=extra_config)
        elif demo_type == 4:
            data = load_othello_model()
            model = data["model"]
            sae = data["sae"]
            tokens = kwargs.get("othello_tokens")
            if tokens is None:
                raise ValueError("Othello tokens must be provided for demo type 4.")
            return generate_feature_centric_vis(model, sae, tokens, output_filename, feature=kwargs.get("feature", 8))
        elif demo_type == 5:
            data = load_attn_model()
            model = data["model"]
            sae = data["sae"]
            tokens = kwargs.get("tokens")
            if tokens is None:
                raise ValueError("Tokens must be provided for demo type 5.")
            return generate_feature_centric_vis(model, sae, tokens, output_filename, feature=kwargs.get("feature", 8))
        else:
            raise ValueError(f"Unsupported demo type: {demo_type}")
    except Exception as e:
        logger.exception("Workflow execution failed for demo type %d.", demo_type)
        raise
