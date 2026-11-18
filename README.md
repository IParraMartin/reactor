# reactor

Human reading times for any text. `reactor` predicts word-level eye-tracking and
self-paced reading measures with a neural model trained on 31 reading corpora in 22
languages.

```python
import reactor as rt

react = rt.NeuralReact()
result = react.predict("the house is red")
print(result)          # 1034.54ms: total reading time of the text
result.words           # every measure for every word
```

## Installation

```bash
pip install reactor-rt
```

- **Requirements.** Python 3.11 or newer, PyTorch 2.6 or newer and Transformers 5.2 or
  newer.
- **First run.** The heads (about 80 MB) download from the
  [Hugging Face Hub](https://huggingface.co/IParraMartin/reactor-qwen3.5-4b). The
  backbone, [Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base) (about
  9 GB), downloads from its own repository. Both are cached in `HF_HOME`, so later runs
  work offline.
- **Hardware.** A CUDA GPU with 16 GB or more is fastest; on smaller cards, lower
  `max_tokens`. Apple silicon (MPS) and CPU also work. On an Apple laptop, loading takes about 20 s and a short sentence about
  0.7 s.
- **Linux GPUs.** `pip install "reactor-rt[cuda]"` also installs `flash-linear-attention`, the fast kernels for the backbone's linear
  attention layers. Without them the layers run in plain PyTorch, which is slower but
  gives the same results up to floating-point precision.

## Usage

```python
import reactor as rt

react = rt.NeuralReact(language="en")            # see the settings below

result = react.predict("The old man the boat.")
result.total                                     # float, milliseconds
result.per_word                                  # the summed measure for every word
result.words                                     # pandas DataFrame, one row per word

# change any setting for one call
react.predict("Das Haus ist rot.", language="de", reader_group="L2")
react.predict("The old man the boat.", paradigm="spr")         # self-paced reading
react.predict("The old man the boat.", measure="gd")           # sum gaze durations
react.predict("The old man the boat.", dataset="provo")        # Provo's readers and setup
react.predict(["我们", "今天", "读", "了", "一", "篇", "文章", "。"], language="zh")

# many texts: batched on the device, same results as one at a time
results = react.predict_batch(["the house is red", "the old man the boat"])
```

`result.words` has one row per word:

| Column | Meaning |
|---|---|
| `word` | The word as given |
| `surprisal` | Surprisal under the backbone, in bits |
| `ffd`, `gd`, `gpt`, `trt` | First-fixation duration, gaze duration, go-past time, total reading time (ms) |
| `fix_prob` | Probability that a reader fixates the word in first pass |
| `reg_prob` | Probability that first-pass reading of the word ends in a regression |
| `rt` | Self-paced reading time (ms), with `paradigm="spr"` instead of the eye-tracking columns |

Predictions are population averages: geometric means over readers. Eye-tracking
durations are for the readers who fixate the word, so `total` counts every word as
fixated. `fix_prob` gives the share of readers who fixate it in first pass.

### Settings

Settings can be given to `NeuralReact(...)` or to any `predict`/`predict_batch` call.

| Setting | Default | Values |
|---|---|---|
| `language` | `"en"` | ISO 639 code; see the supported languages below |
| `paradigm` | `"et"` | `"et"` (eye tracking) or `"spr"` (self-paced reading) |
| `measure` | `"trt"` / `"rt"` | Measure summed into `total`: `trt`, `gd`, `ffd` or `gpt` for eye tracking, `rt` for self-paced reading |
| `condition` | `"reading for comprehension"` | Reading task, as named in training (`rt.model_info().conditions`), e.g. `"information seeking"` |
| `reader_group` | `"L1"` | `"L1"` (native) or `"L2"` (second-language readers) |
| `dataset` | `None` | A training corpus whose readers and setup to predict (`react.datasets`); `None` predicts a generic reader of the language |

The constructor also takes these options:
- `model`: a local directory or Hub repository id; defaults to `$REACTOR_MODEL` or the
  released model.
- `device`: `"cuda"`, `"mps"` or `"cpu"`; the fastest available by default.
- `dtype`: the backbone weight type.
- `revision` and `cache_dir`: the download revision and cache.
- `batch_size` and `max_tokens`: limits for `predict_batch`. Lower `max_tokens` if the
  device runs out of memory.

Text given as a string is split on whitespace, and punctuation stays attached to its
word, as in the training corpora. For languages written without spaces (Chinese,
Japanese, Thai), pass a list of words.

## Supported languages

Trained on Basque, Chinese, Czech, Danish, Dutch, English, Estonian, Finnish, German,
Greek, Hebrew, Hindi, Icelandic, Italian, Korean, Norwegian, Persian, Portuguese,
Russian, Serbian, Spanish and Turkish.
- English has the most data, with 15 corpora.
- Other languages the backbone knows run zero-shot. Their speed comes from the
  paradigm-wide average, and they are untested.

## Results

On held-out test texts of all 31 corpora (342 dataset × condition × reader group ×
language × measure cells):
- The model reaches a mean Pearson r of **0.652** between predicted and observed word
  means.
- That is **78%** of the noise ceiling, i.e. of how well one half of the readers
  predicts the other.

| Measure | r | r / noise ceiling |
|---|---|---|
| Total reading time | 0.762 | 0.87 |
| Gaze duration | 0.693 | 0.83 |
| Go-past time | 0.674 | 0.82 |
| First-fixation duration | 0.453 | 0.63 |
| Fixation probability | 0.821 | 0.88 |
| Regression probability | 0.495 | 0.63 |
| Self-paced reading time | 0.641 | 1.09 |

**Against baselines** on the same data and splits (tier-1 test cells, 95% paired
bootstrap intervals):
- It beats a fine-tuned XLM-RoBERTa-large regressor by +0.051 [+0.049, +0.053].
- It beats a reimplementation of the CMCL 2022 shared-task winner by +0.10 (point
  estimate).
- It beats a gradient-boosting model on the same Qwen3.5-4B surprisal, frequency and
  corpus features by +0.008 [+0.006, +0.011] on the full test set. That margin is
  cell-weighted; dataset-weighted it is +0.006 [−0.003, +0.018], which is not
  significant.
- On corpora left out of training it leads that feature model by +0.014 [+0.009, +0.019].

The [evaluation report](https://huggingface.co/IParraMartin/reactor-qwen3.5-4b/blob/main/docs/evaluation.md) has every comparison and
the known weak spots; the raw tables are in [results/](https://huggingface.co/IParraMartin/reactor-qwen3.5-4b/tree/main/results).

```python
info = rt.model_info()         # bundled with the package, no download
info.datasets_table()          # the 31 training corpora: languages, sizes, licences, citations
info.results_table("per_dataset")
info.training                  # data mix, architecture, optimisation, compute
```

## How it works

1. A prompt states the reading setup. A frozen Qwen3.5-4B-Base then reads the prompt
   and the text.
2. Each word is represented by its last subword token. That representation is mixed
   across all layers and fused with word features: length, frequency, surprisal,
   entropy and position.
3. A dataset embedding conditions the result on the readers and setup. A small
   bidirectional Transformer over the words adds the right context a causal model
   lacks.
4. Three such heads, trained with different seeds, share one backbone and are averaged.

See the [training details](https://huggingface.co/IParraMartin/reactor-qwen3.5-4b/blob/main/docs/training.md) and the
[training corpora](https://huggingface.co/IParraMartin/reactor-qwen3.5-4b/blob/main/docs/datasets.md).

## Limitations

- **Absolute speed.** The model ranks words within a text well, but the overall speed
  of a new population of readers is not identifiable from text. With
  `dataset=None` it predicts a pooled reader of the language. Specific setups can be a
  third or more faster or slower: on unseen PoTeC readers, 439 ms observed against
  289 ms predicted per word. Pass `dataset=` for a known setup, or calibrate on a few
  readers.
- **Weak spots.** First-fixation duration and regression probability are the hardest
  measures, at 63% of the ceiling. The feature baseline does clearly better on Natural
  Stories: 0.78 vs 0.50, the model's lowest ratio to the ceiling.
- **Training domain.** The training texts are mostly sentences and short expository or
  narrative passages. They were read on screen, mostly by young adults.

## Licence

- **Code:** MIT.
- **Model weights:**
  [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), because some
  training corpora are licensed non-commercially.
- **Backbone:** Qwen3.5-4B-Base is Apache 2.0 and is downloaded from its own
  repository.
- [NOTICE.md](https://huggingface.co/IParraMartin/reactor-qwen3.5-4b/blob/main/NOTICE.md) lists the licences of the training corpora.

## Citation

If you use reactor, please cite it ([BibTeX](https://huggingface.co/IParraMartin/reactor-qwen3.5-4b#citation)) and the corpora it was
trained on ([citations](https://huggingface.co/IParraMartin/reactor-qwen3.5-4b/blob/main/docs/datasets.md#citations)).
