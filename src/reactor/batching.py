"""Grouping of sequences into padded batches under a token budget."""

from __future__ import annotations

import numpy as np


def pack_batches(indices: np.ndarray, lengths: np.ndarray, max_tokens: int, max_batch_size: int) -> list[list[int]]:
    """Packs sequences, in the given order, into batches under a padded-token budget.

    A batch is closed as soon as adding the next sequence would make
    ``batch size * longest length`` exceed ``max_tokens`` or the batch would hold more
    than ``max_batch_size`` sequences.

    Args:
        indices: Sequence indices in packing order.
        lengths: Token length of every sequence, indexed by sequence index.
        max_tokens: Maximum ``batch size * longest length`` per batch.
        max_batch_size: Maximum sequences per batch.

    Returns:
        Batches of sequence indices; a sequence longer than the budget gets a batch of
        its own.
    """
    batches: list[list[int]] = []
    current: list[int] = []
    longest = 0
    for index in indices:
        length = int(lengths[index])
        grown = max(longest, length)
        if current and (grown * (len(current) + 1) > max_tokens or len(current) >= max_batch_size):
            batches.append(current)
            current, grown = [], length
        current.append(int(index))
        longest = grown
    if current:
        batches.append(current)
    return batches


def length_sorted_batches(lengths: list[int], max_tokens: int, max_batch_size: int) -> list[list[int]]:
    """Deterministic batches of similar-length sequences covering every sequence once.

    Args:
        lengths: Token length of every sequence.
        max_tokens: Maximum ``batch size * longest length`` per batch.
        max_batch_size: Maximum sequences per batch.

    Returns:
        Batches of sequence indices, shortest sequences first.
    """
    lengths = np.asarray(lengths, dtype="int64")
    order = np.argsort(lengths, kind="stable")
    return pack_batches(order, lengths, max_tokens, max_batch_size)
