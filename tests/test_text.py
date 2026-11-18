from __future__ import annotations

import numpy as np

from reactor.features import strip_punctuation, surface_features
from reactor.text import align_words_to_tokens, conditioning_prompt, join_words, make_windows


def test_join_words_spans_point_at_each_word() -> None:
    text, spans = join_words(["The", "old", "man."], "en")
    assert text == "The old man."
    assert [text[a:b] for a, b in spans] == ["The", "old", "man."]
    joined, chinese_spans = join_words(["我们", "读"], "zh")
    assert joined == "我们读" and chinese_spans == [(0, 2), (2, 3)]


def test_windows_tile_every_word_exactly_once() -> None:
    assert len(make_windows(10, 24, 8)) == 1
    windows = make_windows(100, 24, 8)
    covered = [i for w in windows for i in range(w.target_start, w.target_end)]
    assert covered == list(range(100))
    assert all(w.context_start <= w.target_start and w.target_end <= w.context_end for w in windows)


def test_alignment_finds_first_and_last_token_after_the_prompt() -> None:
    # "ab cd" after a 3-character prompt, tokenized as "ab", " c", "d"
    offsets = np.array([[0, 0], [3, 5], [5, 7], [7, 8]])
    first, last = align_words_to_tokens(offsets, [(0, 2), (3, 5)], 3)
    assert first.tolist() == [1, 2]
    assert last.tolist() == [1, 3]
    truncated_first, truncated_last = align_words_to_tokens(offsets[:2], [(0, 2), (3, 5)], 3)
    assert truncated_last.tolist() == [1, -1]


def test_prompt_and_surface_features() -> None:
    assert conditioning_prompt("de", "spr", "reading", "L2") == (
        "Language: German. Measure: self-paced reading. Task: reading. Readers: non-native speakers.\n\n"
    )
    assert strip_punctuation("«Hello!»") == "Hello"
    features = surface_features(["Hi", "there.", "Bye"], "en", np.full(3, np.nan), np.full(3, -1), np.full(3, -1))
    assert features["sentence_end"].tolist() == [0.0, 1.0, 1.0]
    assert features["sentence_start"].tolist() == [1.0, 0.0, 1.0]
    assert features["line_known"].tolist() == [0.0, 0.0, 0.0]
