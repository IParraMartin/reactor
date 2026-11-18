---
license: cc-by-nc-sa-4.0
base_model: Qwen/Qwen3.5-4B-Base
language:
- cs
- da
- de
- el
- en
- es
- et
- eu
- fa
- fi
- he
- hi
- is
- it
- ko
- nl
- no
- pt
- ru
- sr
- tr
- zh
tags:
- reading-times
- eye-tracking
- self-paced-reading
- psycholinguistics
- surprisal
- cognitive-modeling
---

# reactor-qwen3.5-4b

Word-level human reading times for any text. The model predicts seven
eye-tracking and self-paced reading measures for every word of a text, for a chosen
language, reading task and reader group:
- first-fixation duration, gaze duration, go-past time and total reading time;
- first-pass fixation and regression probabilities;
- self-paced reading time.

It was trained on 31 public reading corpora in 22 languages. It is three trained heads
on a frozen [Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base), averaged at
prediction time.

| | |
|---|---|
| Test score | Pearson r **0.652** between predicted and observed word means, averaged over 342 held-out cells; **78%** of the noise ceiling |
| Languages | Basque, Chinese, Czech, Danish, Dutch, English, Estonian, Finnish, German, Greek, Hebrew, Hindi, Icelandic, Italian, Korean, Norwegian, Persian, Portuguese, Russian, Serbian, Spanish, Turkish |
| Measures | `ffd`, `gd`, `gpt`, `trt`, `fix_prob`, `reg_prob` (eye tracking); `rt` (self-paced reading) |
| Files | `config.json` and `heads/seed{0,1,2}.safetensors` (26 MB each); the backbone is downloaded from its own repository |
| Licence | CC BY-NC-SA 4.0 (weights); the backbone is Apache 2.0 |

## Usage

The model is loaded by the `reactor` Python package (distribution `reactor-rt`, MIT):

```bash
pip install git+https://github.com/IParraMartin/reactor.git
```

```python
import reactor as rt

react = rt.NeuralReact()                # downloads this repository and the backbone once
result = react.predict("the house is red")
print(result)                           # 1034.54ms: total reading time of the text
result.words                            # every measure for every word (pandas DataFrame)

react.predict("Das Haus ist rot.", language="de", reader_group="L2")
react.predict("The old man the boat.", paradigm="spr")        # self-paced reading
react.predict("The old man the boat.", dataset="provo")       # the readers and setup of one corpus
results = react.predict_batch(["the house is red", "the old man the boat"])
```

**Settings.**
- `language`: ISO 639 code.
- `paradigm`: `et` or `spr`.
- `measure`: the measure summed into the total.
- `condition`: the reading task, e.g. `"reading for comprehension"` or
  `"information seeking"`.
- `reader_group`: `L1` or `L2`.
- `dataset`: a training corpus to calibrate to; by default the model predicts a generic
  reader of the language.

**What the predictions are.**
- Predictions are population averages: geometric means over readers.
- Eye-tracking durations are for the readers who fixate the word.
- A CUDA GPU with 16 GB is fastest; Apple silicon and CPU also work.

## Training data

31 corpora: 26 eye-tracking and 5 self-paced reading, with about 725k words of
distinct texts. Corpora include native (L1) and second-language (L2) readers and 16
reading tasks. For every word, each measure was averaged over the corpus's readers.
[docs/datasets.md](docs/datasets.md) lists every corpus with its size, release page,
licence and citation. The corpora are:
- **Multilingual:** MECO L1 (waves 1 and 2) in 13 languages each, and GECO in English
  and Dutch.
- **English:** MECO L2 (waves 1 and 2), OneStop, InteRead, Provo, EMTeC, ZuCo 1.0 and
  2.0, UCL (ET and SPR), Natural Stories and Brown.
- **Chinese:** BSC, BSC II, the Chinese Reading database and HKC.
- **Other languages:** PoTeC (de), CopCo (da), RaCCooNS (nl), TiNT (nl), RastrOS (pt),
  CERVINO (it), RSC (ru), Potsdam-Allahabad (hi), TURead (tr), PSR (fa) and HeCz (cs).

No corpus data or stimulus text is included in this repository.

## Training

- **Inputs.**
  - A plain-text prompt states the reading setup (language, paradigm, task, readers),
    and the text follows.
  - The frozen backbone (revision `1001bb4d`, bfloat16) reads windows of up to 1,024
    tokens.
  - Each word is represented by its last subword token.
- **Head.** Each head has two learned mixes over all 33 hidden-state layers. These are
  fused with 12 word features: length, frequency, backbone surprisal and entropy,
  position, and sentence and line flags. On top come:
  - a dataset embedding (FiLM), dropped 10% of the time to learn a generic reader;
  - a one-layer bidirectional Transformer over the words;
  - linear outputs.
- **Targets and loss.**
  - Log durations are z-scored per dataset, condition and reader group.
  - Durations use a Huber loss and probabilities binary cross-entropy.
  - Words are weighted by n/(n+5) readers.
- **Optimisation.**
  - Datasets are sampled ∝ size^0.5.
  - AdamW at 1e-3 with a cosine schedule and batches of 16k tokens.
  - Early stopping on validation macro r.
- **Ensemble.** Three seeds, about 1.5–2 hours each on one A40. The heads are averaged
  in log space.

[docs/training.md](docs/training.md) has the full recipe.
[results/training/](results/training) has every member's configuration and training
log.

## Evaluation

Scores are on held-out texts (80/10/10 split by text). Pearson r is computed per
dataset × condition × reader group × language × measure cell, then averaged.

| Measure | r | r / noise ceiling |
|---|---|---|
| Total reading time | 0.762 | 0.87 |
| Gaze duration | 0.693 | 0.83 |
| Go-past time | 0.674 | 0.82 |
| First-fixation duration | 0.453 | 0.63 |
| Fixation probability | 0.821 | 0.88 |
| Regression probability | 0.495 | 0.63 |
| Self-paced reading time | 0.641 | 1.09 |

Baselines were run on the same data and splits. The table uses the 239 test cells of
the 12 tier-1 corpora, which every baseline was run on.

| System | Cell macro r | Dataset macro r |
|---|---|---|
| **reactor-qwen3.5-4b** | **0.613** | 0.626 |
| Gradient boosting on word features (incl. Qwen3.5-4B surprisal) + corpus identity | 0.606 | **0.644** |
| Fine-tuned XLM-RoBERTa-large regressor | 0.562 | 0.539 |
| CMCL 2022 multilingual winner, reimplemented (Salicchi et al., 2022) | 0.512 | 0.497 |
| Ridge on frozen-layer representations + surprisal (Tsipidi et al., 2026) | 0.491 | 0.485 |
| Ridge on lexical predictors + surprisal (Tsipidi et al., 2026) | 0.476 | 0.467 |

Paired bootstrap intervals:
- **Fine-tuned XLM-R-large:** +0.051 [+0.049, +0.053].
- **Gradient boosting with corpus identity, full test set:** +0.008 [+0.006, +0.011]
  cell-weighted. Dataset-weighted it is +0.006 [−0.003, +0.018], which is not
  significant.
- **Corpora left out of training (Provo, PoTeC, CopCo):** a model of the same
  architecture scores 0.749 against the feature model's 0.735, +0.014 [+0.009, +0.019].

[docs/evaluation.md](docs/evaluation.md) has the full analysis, and
[results/](results) has every table.

## Limitations

- **Absolute speed.** The model ranks words well, but a new reader population can be
  a third or more faster or slower than the pooled reader it predicts. On unseen PoTeC
  readers it is 439 ms observed against 289 ms predicted per word. Use `dataset=` or
  calibrate on a few readers when absolute milliseconds matter.
- **Weak spots.**
  - First-fixation duration and regression probability are at 63% of the ceiling.
  - Natural Stories is clearly worse than the feature baseline (0.50 vs 0.78).
  - Languages outside the 22 run zero-shot and are untested.
- **Readers and texts.** Readers were mostly young adults reading on screen.
  Predictions are averages over readers, not individual readers.

## Licence

- The weights are released under
  [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Several
  training corpora are licensed non-commercially (Natural Stories, RaCCooNS, TiNT,
  InteRead).
- The backbone, Qwen3.5-4B-Base, is Apache 2.0 and is not redistributed here.
- [NOTICE.md](NOTICE.md) has every corpus licence.

## Citation

Please cite this model and the corpora it was trained on
([docs/datasets.md](docs/datasets.md)).

```bibtex
@software{reactor_2026,
  title = {reactor: human reading times for any text},
  author = {IParraMartin},
  year = {2026},
  version = {0.1.0},
  url = {https://huggingface.co/IParraMartin/reactor-qwen3.5-4b}
}
```
