#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Storage module for the SAE Feature Scouter.

Provides functions to save and load generated HTML outputs.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def save_output(content: str, filename: str, overwrite: bool = True) -> None:
    """
    Save content to a file, optionally overwriting existing files.

    Args:
        content (str): The HTML content to save.
        filename (str): Destination filename.
        overwrite (bool, optional): Whether to overwrite if file exists.
    """
    try:
        path = Path(filename)
        if path.exists() and not overwrite:
            logger.info("File '%s' already exists; skipping save.", filename)
            return
        path.write_text(content, encoding="utf-8")
        logger.info("Content saved successfully to '%s'.", filename)
    except Exception as e:
        logger.exception("Failed to save content to '%s'.", filename)
        raise

def load_output(filename: str) -> str:
    """
    Load content from a file.

    Args:
        filename (str): Source filename.

    Returns:
        str: The file's content.
    """
    try:
        path = Path(filename)
        content = path.read_text(encoding="utf-8")
        logger.info("Content loaded from '%s'.", filename)
        return content
    except Exception as e:
        logger.exception("Failed to load content from '%s'.", filename)
        raise
