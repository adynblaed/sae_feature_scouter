#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data Preparation module for the SAE Feature Scouter.

Provides functions to prepare data for OthelloGPT and Attention-Only models.
"""

import gc
import torch
import logging
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

def prepare_othello_data(device: str) -> tuple:
    """
    Prepare OthelloGPT-related data:
      - Download tokens, target logits, and linear probe files.
      - Run inference to compute 'alive' features.

    Args:
        device (str): Computation device ("cuda" or "cpu").

    Returns:
        tuple: (othello_tokens, othello_target_logits, othello_linear_probes, othello_alive_feats)
    """
    try:
        logger.info("Preparing OthelloGPT data on device '%s'.", device)
        hf_repo_id = "callummcdougall/arena-demos-othellogpt"
        sae_id = "blocks.5.mlp.hook_post-v1"

        def hf_othello_load(filename: str):
            path = hf_hub_download(repo_id=hf_repo_id, filename=filename)
            return torch.load(path, weights_only=True, map_location=device)

        othello_tokens = hf_othello_load("tokens.pt")[:5000]
        othello_target_logits = hf_othello_load("target_logits.pt")[:5000]
        othello_linear_probes = hf_othello_load("linear_probes.pt")
        
        from sae_feature_scouter.services.model_loader import load_othello_model
        data = load_othello_model()
        model = data["model"]
        sae = data["sae"]
        
        hook_name = f"{sae.cfg.hook_name}.hook_sae_acts_post"
        _, cache = model.run_with_cache_with_saes(
            othello_tokens[:128],
            saes=[sae],
            names_filter=hook_name,
        )
        acts = cache[hook_name]
        othello_alive_feats = (acts[:, 5:-5].flatten(0, 1) > 1e-8).any(dim=0).nonzero().squeeze().tolist()
        
        del cache
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("OthelloGPT data prepared successfully.")
        return othello_tokens, othello_target_logits, othello_linear_probes, othello_alive_feats
    except Exception as e:
        logger.exception("Failed to prepare OthelloGPT data.")
        raise

def prepare_attn_data(device: str) -> tuple:
    """
    Prepare data for the attention-only model:
      - Load a streaming dataset.
      - Tokenize and prepend the BOS token.
      - Run inference to compute 'alive' features.

    Args:
        device (str): Computation device ("cuda" or "cpu").

    Returns:
        tuple: (tokens, attn_alive_feats)
    """
    try:
        logger.info("Preparing attention-only model data on device '%s'.", device)
        from sae_feature_scouter.services.model_loader import load_attn_model
        data = load_attn_model()  # Returns dict with keys "model" and "sae"
        model = data["model"]
        sae = data["sae"]

        from sae_feature_scouter.services.data_loader import load_tokens, add_bos_token
        original_dataset = load_tokens(sae.cfg.dataset_path, seq_len=64, batch_size=256, streaming=True)
        bos_token_id = model.tokenizer.bos_token_id
        tokens = add_bos_token(original_dataset, bos_token_id)
        
        hook_name = f"{sae.cfg.hook_name}.hook_sae_acts_post"
        _, cache = model.run_with_cache_with_saes(
            tokens[:64],
            saes=[sae],
            names_filter=hook_name,
            stop_at_layer=sae.cfg.hook_layer + 1,
        )
        acts = cache[hook_name]
        from sae_feature_scouter.services.inference import compute_alive_features
        attn_alive_feats = compute_alive_features(acts)
        
        del cache
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("Attention-only model data prepared successfully.")
        return tokens, attn_alive_feats
    except Exception as e:
        logger.exception("Failed to prepare attention-only model data.")
        raise
