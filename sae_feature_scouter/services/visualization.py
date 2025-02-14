#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualization module for the SAE Feature Scouter.

Provides functions to build visualization configurations and generate HTML outputs.
"""

import logging
from typing import Any, Dict
from sae_vis.data_config_classes import SaeVisConfig
from sae_vis.data_storing_fns import SaeVisData

logger = logging.getLogger(__name__)

def build_vis_config(vis_mode: str, **kwargs: Any) -> SaeVisConfig:
    """
    Build a visualization configuration for the specified mode.

    Args:
        vis_mode (str): Either "feature-centric" or "prompt-centric".
        **kwargs: Additional configuration parameters.

    Returns:
        SaeVisConfig: The configuration object.
    """
    try:
        logger.debug("Building visualization config for mode: '%s'.", vis_mode)
        if vis_mode == "feature-centric":
            config = {"features": kwargs.get("features", list(range(128)))}
            layout = kwargs.get("layout", None)
            if layout is not None:
                config["feature_centric_layout"] = layout
            return SaeVisConfig(**config)
        elif vis_mode == "prompt-centric":
            config = {"features": kwargs.get("features", list(range(128)))}
            return SaeVisConfig(**config)
        else:
            raise ValueError(f"Unknown visualization mode: '{vis_mode}'")
    except Exception as e:
        logger.exception("Failed to build visualization config.")
        raise

def generate_feature_centric_vis(model: Any, sae: Any, tokens: Any, output_filename: str, feature: int, extra_config: Dict[str, Any] = {}) -> str:
    """
    Generate and save a feature-centric visualization.

    Args:
        model (Any): The transformer model.
        sae (Any): The associated SAE.
        tokens (Any): Token data.
        output_filename (str): Destination HTML file.
        feature (int): Feature index to visualize.
        extra_config (Dict[str, Any], optional): Extra config parameters.

    Returns:
        str: The output filename.
    """
    try:
        logger.info("Generating feature-centric visualization for feature %d.", feature)
        cfg = build_vis_config("feature-centric", **extra_config)
        vis_data = SaeVisData.create(sae=sae, model=model, tokens=tokens, cfg=cfg, verbose=True)
        vis_data.save_feature_centric_vis(output_filename, feature=feature)
        logger.info("Feature-centric visualization saved to '%s'.", output_filename)
        return output_filename
    except Exception as e:
        logger.exception("Failed to generate feature-centric visualization.")
        raise

def generate_prompt_centric_vis(model: Any, sae: Any, tokens: Any, prompt: str, seq_pos: int, metric: str, output_filename: str) -> str:
    """
    Generate and save a prompt-centric visualization.

    Args:
        model (Any): The transformer model.
        sae (Any): The associated SAE.
        tokens (Any): Token data.
        prompt (str): Prompt text.
        seq_pos (int): Sequence position.
        metric (str): Metric to use.
        output_filename (str): Destination HTML file.

    Returns:
        str: The output filename.
    """
    try:
        logger.info("Generating prompt-centric visualization for prompt: '%s'.", prompt)
        cfg = build_vis_config("prompt-centric")
        vis_data = SaeVisData.create(sae=sae, model=model, tokens=tokens, cfg=cfg, verbose=True)
        vis_data.save_prompt_centric_vis(output_filename, prompt=prompt, seq_pos=seq_pos, metric=metric)
        logger.info("Prompt-centric visualization saved to '%s'.", output_filename)
        return output_filename
    except Exception as e:
        logger.exception("Failed to generate prompt-centric visualization.")
        raise
