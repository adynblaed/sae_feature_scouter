#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model Loader module for the SAE Feature Scouter.

Provides functions to load transformer models and their corresponding SAEs.
"""

import logging
from typing import Dict, Any
from sae_feature_scouter.services.config import MODEL_CONFIG, DEFAULT_DEVICE

logger = logging.getLogger(__name__)

def load_basic_model(seq_len: int) -> Dict[str, Any]:
    """
    Load the basic 1-layer model and its SAE.

    Args:
        seq_len (int): Sequence length for model input.

    Returns:
        Dict[str, Any]: Dictionary with keys "model", "sae", "sae_B", and "tokens".
    """
    try:
        logger.info("Loading basic model with sequence length %d.", seq_len)
        from sae_vis.model_fns import load_demo_model_saes_and_data
        sae, sae_B, model, all_tokens = load_demo_model_saes_and_data(seq_len, DEFAULT_DEVICE)
        logger.debug("Basic model loaded successfully.")
        return {"model": model, "sae": sae, "sae_B": sae_B, "tokens": all_tokens}
    except Exception as e:
        logger.exception("Failed to load basic model.")
        raise

def load_othello_model() -> Dict[str, Any]:
    """
    Load the OthelloGPT model and its corresponding SAE.
    If the model has no tokenizer, load one via load_othello_vocab and attach a minimal tokenizer that
    defines necessary attributes (e.g. padding_side) and ensures that the vocab values are hashable.

    Returns:
        Dict[str, Any]: Dictionary containing the OthelloGPT model and SAE.
    """
    try:
        logger.info("Loading OthelloGPT model.")
        config = MODEL_CONFIG["othello"]
        from sae_vis.model_fns import load_othello_vocab  # Function to load the vocab
        from sae_lens import SAE, HookedSAETransformer

        # Load the OthelloGPT model.
        othello_model: HookedSAETransformer = HookedSAETransformer.from_pretrained(config["model_name"])

        # If no tokenizer is attached, load it via our helper.
        if not hasattr(othello_model, "tokenizer") or othello_model.tokenizer is None:
            logger.info("No tokenizer found on the OthelloGPT model; loading vocabulary...")
            raw_vocab = load_othello_vocab()  # Expected to return a dict mapping tokens to values.
            
            class SimpleTokenizer:
                def __init__(self, vocab: dict):
                    self.vocab = self.process_vocab(vocab)
                    self.padding_side = "right"  # required by downstream code.
                    self.bos_token = "<bos>"
                    self.eos_token = "<eos>"
                    self.unk_token = "<unk>"
                
                def process_vocab(self, vocab: dict) -> dict:
                    """Ensure all vocab values are hashable."""
                    new_vocab = {}
                    for token, value in vocab.items():
                        if isinstance(value, dict):
                            # If an 'id' key exists, use that.
                            if "id" in value:
                                new_vocab[token] = value["id"]
                            else:
                                # Otherwise, convert the dict into a tuple of sorted items.
                                new_vocab[token] = tuple(sorted(value.items()))
                        else:
                            new_vocab[token] = value
                    return new_vocab
                
                def tokenize(self, text: str):
                    # A simple whitespace tokenizer.
                    return text.split()
            
            tokenizer = SimpleTokenizer(raw_vocab)
            othello_model.tokenizer = tokenizer
            logger.info("Tokenizer loaded with %d tokens.", len(othello_model.tokenizer.vocab))

        othello_sae = SAE.from_pretrained(
            release=config["hf_repo_id"],
            sae_id=config["sae_id"],
            device=DEFAULT_DEVICE
        )[0]
        logger.debug("OthelloGPT model loaded successfully.")
        return {"model": othello_model, "sae": othello_sae}
    except Exception as e:
        logger.exception("Failed to load OthelloGPT model.")
        raise

def load_attn_model() -> Dict[str, Any]:
    """
    Load the attention-only model and its SAE.

    Returns:
        Dict[str, Any]: Dictionary with keys "model" and "sae".
    """
    try:
        logger.info("Loading attention-only model.")
        config = MODEL_CONFIG["attn"]
        from sae_lens import SAE, HookedSAETransformer
        attn_model: HookedSAETransformer = HookedSAETransformer.from_pretrained(config["model_name"])
        attn_sae = SAE.from_pretrained(release=config["hf_repo_id"], sae_id=config["sae_id"], device=DEFAULT_DEVICE)[0]
        logger.debug("Attention-only model loaded successfully.")
        return {"model": attn_model, "sae": attn_sae}
    except Exception as e:
        logger.exception("Failed to load attention-only model.")
        raise
