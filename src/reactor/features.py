"""Word predictors: length, frequency, position, and surprisal from the backbone.

Every word gets the twelve features in ``FEATURE_NAMES``. Surprisal and next-word
entropy come from the frozen base language model. Word probabilities use the
correction of Pimentel and Meister (2024): with tokenizers that attach whitespace to the
*following* word, the probability of a word's tokens is multiplied by the probability
that the next token starts a new word, and divided by the same quantity at the previous
word boundary.
"""

from __future__ import annotations

import logging
import math
import unicodedata

import numpy as np
import torch

from reactor.batching import pack_batches
from reactor.text import NO_SPACE_LANGUAGES, Window, align_words_to_tokens, join_words, make_windows

log = logging.getLogger(__name__)

FEATURE_NAMES = (
    "log_length",
    "zipf",
    "zipf_missing",
    "surprisal",
    "entropy",
    "relative_position",
    "sentence_start",
    "sentence_end",
    "punctuation_end",
    "line_start",
    "line_end",
    "line_known"
)
"""Columns of the feature matrix, in order."""

SENTENCE_FINAL = frozenset(".!?。！？…")
"""Characters that end a sentence when they end a word."""

LN2 = math.log(2.0)
# word 1000 of a document maps to 1
DOCUMENT_POSITION_SCALE = math.log1p(1000.0)


def feature_names(document_position: bool = False) -> tuple[str, ...]:
    """Names of the feature-matrix columns.

    Args:
        document_position: Whether the optional ``document_position`` column is appended.

    Returns:
        ``FEATURE_NAMES``, followed by ``document_position`` when it is enabled.
    """
    return FEATURE_NAMES + (("document_position",) if document_position else ())


def strip_punctuation(word: str) -> str:
    """Removes leading and trailing punctuation and symbols (Unicode categories P and S)."""
    def is_punct(ch: str) -> bool:
        """Whether a character is punctuation or a symbol."""
        return unicodedata.category(ch)[0] in "PS"

    start, end = 0, len(word)
    while start < end and is_punct(word[start]):
        start += 1
    while end > start and is_punct(word[end - 1]):
        end -= 1
    return word[start:end]


def zipf_frequencies(words: list[str], language: str) -> np.ndarray:
    """Looks up word frequencies on the Zipf scale with ``wordfreq``.

    Args:
        words: Words with punctuation attached; it is stripped and the word lowercased.
        language: ISO 639 code.

    Returns:
        Zipf frequency per word (0 for unknown words), or NaN for every word when
        ``wordfreq`` does not support the language or misses its optional tokenizer
        (Chinese, Japanese and Korean need ``wordfreq[cjk]``).
    """
    from wordfreq import zipf_frequency

    try:
        zipf_frequency("a", language)
    except (LookupError, ValueError, ImportError) as error:
        log.warning("no wordfreq frequencies for %s: %s", language, error)
        return np.full(len(words), np.nan)
    return np.array([zipf_frequency(strip_punctuation(w).lower(), language) for w in words])


def surface_features(
    words: list[str],
    language: str,
    sentence_idx: np.ndarray,
    line_start: np.ndarray,
    line_end: np.ndarray
) -> dict[str, np.ndarray]:
    """Computes the predictors that need no language model.

    Sentence boundaries come from ``sentence_idx`` when it is known for every word, and
    otherwise from sentence-final punctuation, with the last word always ending a
    sentence.

    Args:
        words: Words of one text.
        language: ISO 639 code.
        sentence_idx: Sentence index per word (NaN when unknown).
        line_start: 1 if the word starts a screen line, 0 if not, -1 if unknown.
        line_end: 1 if the word ends a screen line, 0 if not, -1 if unknown.

    Returns:
        Arrays keyed by feature name, all of ``FEATURE_NAMES`` except ``surprisal`` and
        ``entropy``.
    """
    n = len(words)
    stripped = [strip_punctuation(w) for w in words]
    zipf = zipf_frequencies(words, language)
    punctuation_end = np.array([float(bool(w) and w[-1] in SENTENCE_FINAL) for w in words])
    sentence_idx = np.asarray(sentence_idx, dtype="float64")
    if np.isfinite(sentence_idx).all():
        sentence_start = np.r_[1.0, (np.diff(sentence_idx) != 0).astype(float)]
        sentence_end = np.r_[(np.diff(sentence_idx) != 0).astype(float), 1.0]
    else:
        sentence_end = punctuation_end.copy()
        sentence_end[-1] = 1.0
        sentence_start = np.r_[1.0, sentence_end[:-1]]
    line_start = np.asarray(line_start, dtype="float64")
    line_end = np.asarray(line_end, dtype="float64")
    return {
        "log_length": np.log1p([len(s) for s in stripped]),
        "zipf": np.nan_to_num(zipf, nan=0.0) / 8.0,
        "zipf_missing": np.isnan(zipf).astype(float),
        "relative_position": np.arange(n) / max(n - 1, 1),
        "sentence_start": sentence_start,
        "sentence_end": sentence_end,
        "punctuation_end": punctuation_end,
        "line_start": np.clip(line_start, 0, 1),
        "line_end": np.clip(line_end, 0, 1),
        "line_known": (line_start >= 0).astype(float)
    }


def feature_matrix(
    words: list[str],
    language: str,
    sentence_idx: np.ndarray,
    line_start: np.ndarray,
    line_end: np.ndarray,
    surprisal: np.ndarray,
    entropy: np.ndarray,
    document_index: np.ndarray | None = None
) -> np.ndarray:
    """Stacks all word predictors into the model's feature matrix.

    Surprisal and entropy are divided by 10 bits so that every feature is roughly unit
    scale. ``document_index`` adds the optional ``document_position`` feature,
    ``log(1 + index)`` scaled so that word 1000 maps to 1, for models trained with it.

    Args:
        words: Words of one text.
        language: ISO 639 code.
        sentence_idx: Sentence index per word (NaN when unknown).
        line_start: Line-start flags (-1 unknown).
        line_end: Line-end flags (-1 unknown).
        surprisal: Word surprisal in bits.
        entropy: Next-word entropy in bits.
        document_index: Position of each word in its document; None leaves out
            ``document_position``.

    Returns:
        Float32 array of shape ``(n_words, len(feature_names(document_index is not None)))``.
    """
    columns = surface_features(words, language, sentence_idx, line_start, line_end)
    columns["surprisal"] = np.nan_to_num(np.asarray(surprisal, dtype="float64"), nan=0.0) / 10.0
    columns["entropy"] = np.nan_to_num(np.asarray(entropy, dtype="float64"), nan=0.0) / 10.0
    if document_index is not None:
        index = np.clip(np.asarray(document_index, dtype="float64"), 0, None)
        columns["document_position"] = np.log1p(index) / DOCUMENT_POSITION_SCALE
    names = feature_names(document_index is not None)
    return np.stack([columns[name] for name in names], axis=1).astype("float32")


def word_start_token_mask(tokenizer, vocab_size: int) -> torch.Tensor:
    """Marks the vocabulary entries that begin a new whitespace-separated word.

    Args:
        tokenizer: A Hugging Face tokenizer.
        vocab_size: Size of the model's output layer, which may exceed the tokenizer's
            vocabulary because of padding.

    Returns:
        Boolean tensor of shape ``(vocab_size,)``; end-of-sequence tokens count as word
        starts because they also close the previous word.
    """
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    pieces = tokenizer.convert_ids_to_tokens(list(range(min(vocab_size, len(tokenizer)))))
    for i, piece in enumerate(pieces):
        if piece is None:
            continue
        if piece[:1] in ("Ġ", "▁", "Ċ", "ĉ") or piece[:1].isspace():
            mask[i] = True
    if tokenizer.eos_token_id is not None:
        mask[tokenizer.eos_token_id] = True
    return mask


@torch.no_grad()
def lm_word_features(
    decoder: torch.nn.Module,
    lm_head: torch.nn.Module,
    tokenizer,
    texts: list[list[str]],
    language: str,
    word_start_mask: torch.Tensor,
    target_words: int = 384,
    context_words: int = 192,
    max_tokens: int = 4096,
    max_batch_size: int = 32
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Computes word surprisal and next-word entropy, in bits, for several texts.

    Each text is read on its own: long texts are processed in windows that keep
    ``context_words`` words of left context, and the windows of all texts are sorted by
    length and run through the decoder in right-padded batches. The decoder is causal,
    so padding after a window leaves its outputs unchanged, and an unpadded batch runs
    without an attention mask exactly as a single window would. Vocabulary
    log-probabilities are computed one window at a time to bound memory. The
    beginning-of-word correction applies only to languages written with spaces.

    Args:
        decoder: Frozen transformer returning ``last_hidden_state``.
        lm_head: Output projection to vocabulary logits.
        tokenizer: Matching fast tokenizer.
        texts: Words of every text.
        language: ISO 639 code shared by the texts.
        word_start_mask: Output of ``word_start_token_mask``.
        target_words: Words scored per window.
        context_words: Words of left context per window.
        max_tokens: Maximum ``batch size * longest window`` in tokens per decoder call.
        max_batch_size: Maximum windows per decoder call.

    Returns:
        ``(surprisal, entropy)`` arrays for every text, in input order; NaN for words
        that received no token.
    """
    device = next(decoder.parameters()).device
    start_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else start_id
    correct = language not in NO_SPACE_LANGUAGES
    results = [(np.full(len(words), np.nan), np.full(len(words), np.nan)) for words in texts]
    bow_mask = word_start_mask.to(device)
    windows = []
    for t, words in enumerate(texts):
        for window in make_windows(len(words), target_words, context_words):
            chunk = words[window.context_start:window.target_end]
            text, spans = join_words(chunk, language)
            encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
            offsets = np.asarray(encoding["offset_mapping"], dtype="int64").reshape(-1, 2)
            first, last = align_words_to_tokens(offsets, spans, 0)
            windows.append((t, window, [start_id, *encoding["input_ids"]], first, last))
    lengths_all = np.array([len(w[2]) for w in windows], dtype="int64")
    order = np.argsort(lengths_all, kind="stable")
    for batch in pack_batches(order, lengths_all, max_tokens, max_batch_size):
        lengths = [int(lengths_all[i]) for i in batch]
        input_ids = torch.full((len(batch), max(lengths)), pad_id, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for row, i in enumerate(batch):
            input_ids[row, :lengths[row]] = torch.tensor(windows[i][2])
            attention_mask[row, :lengths[row]] = 1
        padded = min(lengths) < max(lengths)
        hidden = decoder(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device) if padded else None,
            use_cache=False
        ).last_hidden_state
        for row, i in enumerate(batch):
            t, window, ids, first, last = windows[i]
            log_probs = torch.log_softmax(lm_head(hidden[row, :lengths[row]]).float(), dim=-1)
            score_window(log_probs, torch.tensor(ids, device=device), first, last, window, correct, bow_mask, *results[t])
    return results


def score_window(
    log_probs: torch.Tensor,
    input_ids: torch.Tensor,
    first: np.ndarray,
    last: np.ndarray,
    window: Window,
    correct: bool,
    bow_mask: torch.Tensor,
    surprisal: np.ndarray,
    entropy: np.ndarray
) -> None:
    """Writes the surprisal and entropy of a window's target words into a text's arrays.

    A word's surprisal is the negative log-probability of the tokens it owns, from its
    first token not shared with the previous word to its last. With ``correct``, the
    log-probability that the token after the word starts a new word is subtracted, and
    the same quantity after the previous word is added back. Entropy is that of the
    next-token distribution at the word's first owned token.

    Args:
        log_probs: Next-token log-probabilities, ``(tokens, vocab)``, for the window
            including its start token.
        input_ids: Token ids of the window, start token first.
        first: First token of every window word, counted without the start token.
        last: Last token of every window word; -1 if the word has no token.
        window: The window, whose word indices refer to the whole text.
        correct: Apply the beginning-of-word correction.
        bow_mask: Vocabulary mask of word-initial tokens.
        surprisal: The text's surprisal array, filled in place.
        entropy: The text's entropy array, filled in place.
    """
    token_lp = log_probs[:-1].gather(1, input_ids[1:, None])[:, 0].cpu().numpy()
    bow_lp = torch.logsumexp(log_probs.masked_fill(~bow_mask, -torch.inf), dim=-1).cpu().numpy()
    step_h = (-(log_probs.exp() * log_probs).nan_to_num().sum(-1)).cpu().numpy()
    previous_last = -1
    for i in range(len(first)):
        if last[i] < 0:
            continue
        owned_first = max(first[i], previous_last + 1)
        value = -token_lp[owned_first:last[i] + 1].sum()
        if correct:
            value -= bow_lp[last[i] + 1]
            if previous_last >= 0:
                value += bow_lp[previous_last + 1]
        word_index = window.context_start + i
        if window.target_start <= word_index < window.target_end:
            surprisal[word_index] = value / LN2
            entropy[word_index] = step_h[owned_first] / LN2
        previous_last = last[i]
