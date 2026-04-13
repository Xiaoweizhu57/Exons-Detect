"""Utility helpers for Exons-Detect."""

from __future__ import annotations

from transformers import AutoTokenizer


def assert_tokenizer_consistency(model_id_1: str, model_id_2: str) -> None:
    """Raise if two model identifiers resolve to different vocabularies."""
    tokenizer_1 = AutoTokenizer.from_pretrained(model_id_1)
    tokenizer_2 = AutoTokenizer.from_pretrained(model_id_2)

    if tokenizer_1.get_vocab() != tokenizer_2.get_vocab():
        raise ValueError(
            f"Tokenizers are not identical for '{model_id_1}' and '{model_id_2}'."
        )
