# Evaluation

All numbers are on held-out texts that no model saw in training or model selection.
The full tables are in [`results/`](../results); `reactor.model_info().results` serves
the same summaries.

## How the model is scored

- **Cells.** A cell is one combination of dataset, reading condition, reader group,
  language and measure, e.g. MECO wave 1, German, native readers, gaze duration. There
  are 342 test cells across the 31 corpora.
- **Metric.** In each cell, Pearson r between predicted and observed word means. Each
  observed mean is taken over the readers who read the word: in log space for
  durations, as a share of readers for probabilities.
- **Averages.** *Cell macro* averages r over cells. *Dataset macro* averages over cells
  within each dataset first, then over datasets, so MECO's 156 cells don't dominate.
- **Noise ceiling.** The readers of every text are split into two random halves 20
  times. The two half-means are correlated and Spearman-Brown corrected, and the square
  root of that reliability bounds the r any model can reach. `pearson_normalized` is r
  divided by this ceiling.
  - With few readers per word the estimate is noisy and can fall below the model's r.
    This happens for UCL self-paced reading and the pooled `rt` cells, where the ratio
    exceeds 1.
- **Uncertainty.** Model comparisons use a paired bootstrap (2,000 samples). It
  resamples whole texts within each dataset, condition, reader group and language.
  - The test fold has a single text for each MECO language and for PoTeC, so their
    per-dataset intervals reflect no text sampling.
- **Splits.**
  - 80/10/10 by text within each dataset.
  - MECO is split by item, so translations of a text share a split; OneStop is split by
    article, so its difficulty levels share a split.
  - The split seed was fixed before any model was trained.

## Overall

| Split | Cells | Cell macro r | Dataset macro r | r / ceiling |
|---|---|---|---|---|
| Validation | 342 | 0.657 | 0.706 | 0.789 |
| **Test** | 342 | **0.652** | **0.690** | **0.783** |

The three ensemble members alone score 0.643 to 0.644 cell macro on test
([`results/training/members.csv`](../results/training/members.csv)), so ensembling adds
about +0.008.

## Per measure (test)

| Measure | Cells | Pearson r | Ceiling r | r / ceiling | MAE |
|---|---|---|---|---|---|
| First-fixation duration (`ffd`) | 56 | 0.453 | 0.730 | 0.63 | 20.8 ms |
| Gaze duration (`gd`) | 56 | 0.693 | 0.837 | 0.83 | 31.7 ms |
| Go-past time (`gpt`) | 54 | 0.674 | 0.821 | 0.82 | 60.0 ms |
| Total reading time (`trt`) | 60 | 0.762 | 0.873 | 0.87 | 47.7 ms |
| First-pass fixation probability (`fix_prob`) | 56 | 0.821 | 0.936 | 0.88 | 0.105 |
| First-pass regression probability (`reg_prob`) | 54 | 0.495 | 0.791 | 0.63 | 0.117 |
| Self-paced reading time (`rt`) | 6 | 0.641 | 0.692 | 1.09 | 21.6 ms |

The late, integrative measures (total reading time, gaze duration, go-past time) are
close to their ceilings. First-fixation duration and regression probability have the
most room left.

## Baselines (tier-1 test cells)

All systems were trained on the same data with the same splits and metric. The table
uses the 239 test cells of the 12 tier-1 corpora, the subset every baseline was run on
([`results/benchmark.csv`](../results/benchmark.csv)).

| System | Cell macro r | Dataset macro r | TRT MAE |
|---|---|---|---|
| **reactor (this release)** | **0.613** | 0.626 | 49.1 ms |
| Gradient boosting on word features + corpus identity (tier-2 data) | 0.606 | **0.644** | 52.7 ms |
| Gradient boosting on word features (tier-1 data) | 0.590 | 0.596 | 59.9 ms |
| Fine-tuned XLM-RoBERTa-large regressor | 0.562 | 0.539 | 53.5 ms |
| CMCL 2022 multilingual winner, reimplemented (HkAmsters; Salicchi et al., 2022) | 0.512 | 0.497 | 55.4 ms |
| Ridge on frozen-layer representations + surprisal (Tsipidi et al., 2026) | 0.491 | 0.485 | 55.9 ms |
| Ridge on lexical predictors + surprisal (Tsipidi et al., 2026) | 0.476 | 0.467 | 51.9 ms |
| *Per-corpus blend of reactor and gradient boosting (not released)* | *0.627* | *0.662* | *48.1 ms* |

The gradient-boosting baselines are strong because they use this project's feature
pipeline:
- Qwen3.5-4B surprisal and entropy, the same frozen language model as reactor.
- Word length, frequency and position.
- In the tier-2 version, the identity of the corpus.

## Paired comparisons

Differences are reactor minus the other system, with 95% bootstrap intervals
([`results/comparisons.csv`](../results/comparisons.csv)).

| Comparison | Cells | Cell macro | Dataset macro |
|---|---|---|---|
| vs fine-tuned XLM-R-large (tier-1 test) | 239 | +0.051 [+0.049, +0.053] | +0.087 [+0.075, +0.098] |
| vs gradient boosting + corpus (full tier-2 test) | 342 | +0.008 [+0.006, +0.011] | +0.006 [−0.003, +0.018] |

**Against gradient boosting + corpus, by measure.**
- reactor is ahead on every duration: `ffd` +0.005, `gd` +0.011, `gpt` +0.021 and
  `trt` +0.013, all with intervals above zero.
- It is also ahead on `reg_prob`, by +0.009.
- It is behind on `fix_prob`, by −0.012 [−0.014, −0.009].

**Against gradient boosting + corpus, by corpus.**
- reactor is clearly better on GECO, MECO L1 wave 2, PoTeC, RastrOS, Chinese Reading,
  Brown, TiNT and UCL self-paced reading.
- It is clearly worse on Natural Stories: 0.50 vs 0.78, one cell, interval
  [−0.45, +0.01].
- It is also worse on Hindi: 0.57 vs 0.63, though that interval includes zero.
- Most of the Natural Stories gap is position in the story. The texts are cut into
  passages for the splits. Readers are much slower at a story's opening passage, and
  the feature model's corpus and position features capture that.

## Generalisation to unseen corpora

To measure what happens on new text, a model with the same architecture was trained on
the tier-1 corpora except Provo, PoTeC and CopCo, and then scored on all texts of those
three ([`results/heldout.csv`](../results/heldout.csv),
[`results/heldout_comparison.csv`](../results/heldout_comparison.csv)).

| Corpus | reactor architecture | Gradient boosting on word features |
|---|---|---|
| Provo (en) | 0.787 | 0.792 |
| PoTeC (de) | 0.780 | 0.749 |
| CopCo (da) | 0.542 | 0.520 |
| **All (14 cells)** | **0.749** | 0.735 |

The neural model is ahead by +0.014 [+0.009, +0.019] on corpora it has never seen.

**Absolute speed.** The model ranks words well in unseen corpora, but it cannot know
how fast a new corpus's readers are.
- Median total reading time on Provo is 237 ms observed against 238 ms predicted.
- PoTeC readers are slower than the pooled German reader the model predicts: 439 ms
  observed against 289 ms predicted.
- CopCo native readers are faster: 253 ms observed against 392 ms predicted.

To get absolute milliseconds for a specific setup, calibrate the level on a few
readers' data. You can also pass `dataset=` to predict for the readers of a training
corpus.
