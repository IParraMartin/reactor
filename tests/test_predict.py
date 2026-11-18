from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from conftest import make_model_dir
from reactor import NeuralReact, Prediction, model_info
from reactor.backbone import load_backbone
from reactor.features import lm_word_features, word_start_token_mask
from reactor.hub import resolve_model
from reactor.model import ModelConfig

TEXT = "the reader slowly reads surprising sentences"


def batch_texts() -> list:
    """Texts of very different lengths; the long one spans several windows."""
    rng = np.random.default_rng(0)
    vocabulary = "the a reader slowly reads surprising sentences about old boats and red houses".split()
    return [
        TEXT,
        "a reader",
        " ".join(rng.choice(vocabulary, size=120)),
        "red",
        "the old man the boat and the reader reads it twice".split()
    ]


def test_prediction_reports_every_measure_and_the_total(model_dir):
    react = NeuralReact(model_dir, device="cpu")
    result = react.predict(TEXT)
    assert isinstance(result, Prediction)
    assert list(result.words.columns) == ["word", "surprisal", "ffd", "gd", "gpt", "trt", "fix_prob", "reg_prob"]
    assert list(result.words["word"]) == TEXT.split()
    assert result.measure == "trt"
    assert result.per_word == result.words["trt"].tolist()
    assert result.total == pytest.approx(result.words["trt"].sum())
    assert str(result) == f"{result.total:.2f}ms"
    assert float(result) == pytest.approx(result.total)
    assert (result.words[["ffd", "gd", "gpt", "trt"]] > 0).all().all()
    assert result.words[["fix_prob", "reg_prob"]].stack().between(0, 1).all()
    assert react.predict(TEXT).total == pytest.approx(result.total)


def test_ensemble_is_the_geometric_mean_of_its_members(dense_backbone_dir, model_dir, tmp_path):
    ensemble = NeuralReact(model_dir, device="cpu").predict(TEXT).words
    singles = [
        NeuralReact(make_model_dir(dense_backbone_dir, tmp_path / f"seed{seed}", (seed,)), device="cpu").predict(TEXT).words
        for seed in (0, 1)
    ]
    assert not np.allclose(singles[0]["trt"], singles[1]["trt"])
    geometric = np.sqrt(singles[0]["trt"] * singles[1]["trt"])
    assert np.allclose(ensemble["trt"], geometric, rtol=1e-5)
    assert np.allclose(ensemble["fix_prob"], (singles[0]["fix_prob"] + singles[1]["fix_prob"]) / 2, atol=1e-6)


@pytest.mark.parametrize("fixture_name", ["model_dir", "hybrid_model_dir"])
def test_batches_match_one_text_at_a_time(fixture_name, request):
    react = NeuralReact(request.getfixturevalue(fixture_name), device="cpu", batch_size=4, max_tokens=2048)
    texts = batch_texts()
    batched = react.predict_batch(texts)
    assert [p.text for p in batched] == texts
    for text, prediction in zip(texts, batched):
        single = react.predict(text).words
        assert list(prediction.words["word"]) == list(single["word"])
        assert prediction.words["trt"].notna().all()
        for column in single.columns[1:]:
            assert np.allclose(prediction.words[column], single[column], rtol=1e-4, atol=1e-4), column
    assert react.predict_batch([]) == []


def test_surprisal_batches_match_single_texts(hybrid_backbone_dir):
    backbone = load_backbone(str(hybrid_backbone_dir), "main", "float32")
    mask = word_start_token_mask(backbone.tokenizer, backbone.lm_head.weight.shape[0])
    texts = [t.split() if isinstance(t, str) else t for t in batch_texts()]
    with torch.no_grad():
        batched = lm_word_features(backbone.decoder, backbone.lm_head, backbone.tokenizer, texts, "en", mask, max_tokens=1024)
        for words, (surprisal, entropy) in zip(texts, batched):
            single = lm_word_features(backbone.decoder, backbone.lm_head, backbone.tokenizer, [words], "en", mask, max_tokens=1, max_batch_size=1)[0]
            assert np.isfinite(surprisal).all() and (surprisal >= 0).all()
            assert np.allclose(surprisal, single[0], atol=1e-4)
            assert np.allclose(entropy, single[1], atol=1e-4)


def test_settings_overrides_and_validation(model_dir):
    react = NeuralReact(model_dir, device="cpu")
    assert react.datasets == ["toy"]
    spr = react.predict("a reader skips words", paradigm="spr")
    assert spr.measure == "rt" and list(spr.words.columns) == ["word", "surprisal", "rt"]
    assert react.predict(TEXT, measure="gd").total == pytest.approx(react.predict(TEXT).words["gd"].sum())
    calibrated = react.predict(TEXT, dataset="toy").words["trt"]
    assert not np.allclose(calibrated, react.predict(TEXT).words["trt"])
    assert react.predict(["我们", "读", "文章"], language="zh").words["word"].tolist() == ["我们", "读", "文章"]
    with pytest.raises(ValueError):
        react.predict("我们读文章", language="zh")
    with pytest.raises(ValueError):
        react.predict(TEXT, measure="rt")
    with pytest.raises(ValueError):
        react.predict(TEXT, dataset="unknown")
    with pytest.raises(ValueError):
        react.predict("   ")
    with pytest.raises(TypeError):
        react.predict(TEXT, lang="en")
    with pytest.raises(TypeError):
        react.predict_batch(TEXT)
    with pytest.raises(ValueError):
        NeuralReact(model_dir, device="cpu", reader_group="native")


def test_model_files_are_checked(model_dir, tmp_path):
    assert resolve_model(model_dir) == model_dir
    with pytest.raises(FileNotFoundError):
        resolve_model(tmp_path)
    raw = json.loads((model_dir / "config.json").read_text())
    (tmp_path / "config.json").write_text(json.dumps({**raw, "format_version": 99}))
    with pytest.raises(ValueError, match="newer reactor"):
        ModelConfig.from_file(tmp_path / "config.json")
    (tmp_path / "config.json").write_text(json.dumps({**raw, "format": "other"}))
    with pytest.raises(ValueError, match="not a reactor model"):
        ModelConfig.from_file(tmp_path / "config.json")


def test_packaged_model_info_describes_the_release():
    info = model_info()
    assert info.name.endswith("reactor-qwen3.5-4b")
    assert len(info.datasets) == 31
    assert {"en", "de", "zh", "hi", "ru"} <= set(info.languages)
    assert "reading for comprehension" in info.conditions
    assert 0.5 < info.results["test"]["cell_macro"] < 1.0
    assert len(info.datasets_table()) == 31
