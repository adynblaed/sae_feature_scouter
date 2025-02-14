#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inference module for the SAE Feature Scouter.

Provides functions to run model inference with caching and extract 'alive' SAE features.
"""

import logging
from typing import Any, Dict, List, Optional
import torch

logger = logging.getLogger(__name__)

def run_inference(model: Any, tokens: torch.Tensor, saes: List[Any], hook_name: str, stop_at_layer: Optional[int] = None) -> Dict[str, torch.Tensor]:
    """
    Execute a forward pass on the model with cache capturing for specified SAEs.

    Args:
        model (Any): The transformer model.
        tokens (torch.Tensor): Input tokens to process.
        saes (List[Any]): List of SAE modules to hook.
        hook_name (str): Name of the hook for capturing activations.
        stop_at_layer (Optional[int], optional): Layer to stop the forward pass.

    Returns:
        Dict[str, torch.Tensor]: Dictionary mapping hook names to captured activations.
    """
    try:
        logger.info("Running inference with hook '%s'.", hook_name)
        run_args: Dict[str, Any] = {"saes": saes, "names_filter": hook_name}
        if stop_at_layer is not None:
            run_args["stop_at_layer"] = stop_at_layer
        _, cache = model.run_with_cache_with_saes(tokens, **run_args)
        logger.debug("Inference completed and activations captured.")
        return cache
    except Exception as e:
        logger.exception("Error during model inference.")
        raise

def compute_alive_features(acts: torch.Tensor, threshold: float = 1e-8, slice_range: Optional[tuple] = None) -> List[int]:
    """
    Compute the indices of 'alive' features from the activation tensor.

    Args:
        acts (torch.Tensor): Activation tensor.
        threshold (float, optional): Threshold for a feature to be considered 'alive'.
        slice_range (Optional[tuple], optional): Tuple (start, end) for slicing tokens.

    Returns:
        List[int]: List of indices corresponding to alive features.
    """
    try:
        if slice_range is not None:
            acts = acts[:, slice_range[0]:slice_range[1]]
        alive_features = (acts.flatten(0, 1) > threshold).any(dim=0).nonzero().squeeze().tolist()
        if isinstance(alive_features, int):
            alive_features = [alive_features]
        logger.debug("Computed alive features: %s", str(alive_features))
        return alive_features
    except Exception as e:
        logger.exception("Error computing alive features.")
        raise
