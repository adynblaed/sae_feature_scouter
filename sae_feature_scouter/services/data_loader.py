#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data Loader module for the SAE Feature Scouter.

Provides functions to load datasets and tokenize input sequences.
"""

import logging
from typing import Optional
import torch
from datasets import load_dataset
from sae_feature_scouter.services.config import DEFAULT_BATCH_SIZE

logger = logging.getLogger(__name__)

def load_tokens(dataset_path: str, seq_len: int, batch_size: int = DEFAULT_BATCH_SIZE, streaming: bool = False) -> torch.Tensor:
    """
    Load and tokenize input sequences from a dataset.

    Args:
        dataset_path (str): Path or identifier for the dataset.
        seq_len (int): Desired sequence length.
        batch_size (int, optional): Number of examples to load. Defaults to DEFAULT_BATCH_SIZE.
        streaming (bool, optional): Whether to load the dataset in streaming mode.

    Returns:
        torch.Tensor: A tensor containing the tokenized sequences.

    Raises:
        Exception: If loading fails.
    """
    try:
        logger.info("Loading dataset from '%s' (streaming=%s) with batch size %d", dataset_path, streaming, batch_size)
        dataset = load_dataset(dataset_path, split="train", streaming=streaming)
        seq_list = []
        for i, example in enumerate(dataset):
            if i >= batch_size:
                break
            seq_list.append(example["input_ids"][: seq_len - 1])
        tokens = torch.tensor(seq_list)
        logger.info("Loaded %d token sequences.", tokens.size(0))
        return tokens
    except Exception as e:
        logger.exception("Failed to load tokens from dataset '%s'.", dataset_path)
        raise

def add_bos_token(tokens: torch.Tensor, bos_token_id: int) -> torch.Tensor:
    """
    Prepend a beginning-of-sequence (BOS) token to each sequence.

    Args:
        tokens (torch.Tensor): Tensor of shape (batch_size, seq_length).
        bos_token_id (int): Identifier for the BOS token.

    Returns:
        torch.Tensor: Token tensor with BOS prepended.

    Raises:
        Exception: If token concatenation fails.
    """
    try:
        batch_size = tokens.shape[0]
        bos_tensor = torch.full((batch_size, 1), bos_token_id, dtype=tokens.dtype, device=tokens.device)
        tokens_with_bos = torch.cat([bos_tensor, tokens], dim=1)
        logger.debug("Prepended BOS token (id=%d) to %d sequences.", bos_token_id, batch_size)
        return tokens_with_bos
    except Exception as e:
        logger.exception("Error while adding BOS token.")
        raise
