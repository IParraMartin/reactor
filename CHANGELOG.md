# Changelog

## 0.1.0 (2026-11-17)

First release.

- `NeuralReact`: word-level eye-tracking and self-paced reading predictions for any text,
  with `predict` and batched `predict_batch`.
- The `reactor-qwen3.5-4b` model, v0.1.0 on the Hugging Face Hub, downloaded on first
  use:
  - three heads on a frozen Qwen3.5-4B-Base;
  - trained on 31 corpora in 22 languages;
  - test r 0.652.
- `model_info()`: the model's training data, languages, recipe and results, bundled
  with the package.
