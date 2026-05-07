from __future__ import annotations

import logging
from typing import List

log = logging.getLogger(__name__)

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model = None


def _get_model():
    global _model
    if _model is None:
        from fastembed import TextEmbedding
        log.info("Loading embedding model %s (first use)...", _MODEL_NAME)
        _model = TextEmbedding(_MODEL_NAME)
        log.info("Embedding model loaded.")
    return _model


def embed(text: str) -> List[float]:
    """Embed a single text string → 384-dim float list."""
    model = _get_model()
    vectors = list(model.embed([text]))
    return vectors[0].tolist()
