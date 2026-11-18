# How the model was trained

The model was trained with the research code in the separate training repository; this
package contains only what is needed for prediction. The exact configuration of each
ensemble member is in [`results/training/`](../results/training), along with its
training log (loss per measure, gradient norm, learning rate, elapsed time and memory,
every 20 steps).

## Targets

The training data contains, for every word of every text in the 31 corpora (see
[datasets.md](datasets.md)), the population average of seven reading measures:

| Measure | Definition |
|---|---|
| `ffd` | First-fixation duration: the first fixation on the word in first pass |
| `gd` | Gaze duration: all first-pass fixations on the word before leaving it |
| `gpt` | Go-past time: from first entering the word until moving past it to the right, including regressions to earlier words |
| `trt` | Total reading time: all fixations on the word |
| `fix_prob` | Share of readers who fixate the word in first pass |
| `reg_prob` | Share of first-pass readers who regress out of the word |
| `rt` | Self-paced reading time of the word |

- **Durations.** Each duration is averaged in log space (a geometric mean) over the
  readers who have it. For eye-tracking durations these are the readers who fixated the
  word.
- **Normalization.** Log durations are z-scored per dataset, condition and reader
  group. The statistics ship with the model to map predictions back to milliseconds.
  - Pooled statistics per language, paradigm and reader group give the *generic
    reader* used when no training corpus is named.
- **Unifying corpora.** Corpora that ship only fixation sequences have the measures
  recomputed from fixations with one shared definition. Measures a corpus does not
  provide are masked out of the loss.

## Inputs

1. **Prompt.** A prefix states the reading setup, e.g. `Language: German. Measure:
   eye-tracking. Task: reading for comprehension. Readers: native speakers.` The text
   follows. Words are joined with spaces, or without them for Chinese.
2. **Windows.** Texts of up to 384 words are one window of at most 1,024 tokens. Longer
   texts are split into windows that predict 256 words each, with up to 64 words of
   context on each side.
3. **Backbone.** The frozen
   [Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base) (revision `1001bb4d`,
   bfloat16) reads each window. It is a base model because instruction tuning worsens
   how well language models fit human reading times (Kuribayashi et al., 2024).
   - A word is represented by its last subword token, the first one that has seen the
     whole word.
4. **Word features.** Twelve per word:
   - `log_length`: log character length.
   - `zipf` and `zipf_missing`: wordfreq Zipf frequency, and a flag for words it does
     not know.
   - `surprisal` and `entropy`: from the same frozen backbone, in bits. Surprisal uses
     the leading-whitespace correction of Pimentel & Meister (2024).
   - `relative_position`: position within the text.
   - `sentence_start`, `sentence_end`, `punctuation_end`: sentence and punctuation
     flags.
   - `line_start`, `line_end`, `line_known`: line flags. These are known only for
     corpora with screen coordinates and are set to unknown at prediction time.

## Head

Each ensemble member is a 26 MB head on the frozen backbone:

1. Two learned softmax mixes over all 33 hidden-state layers (embeddings plus 32 blocks),
   so early-pass and late-pass measures can draw on different depths.
2. A projection to 512 dimensions, fused with an embedding of the word features.
3. A dataset embedding that scales and shifts the result (FiLM). It is replaced by a
   generic-reader embedding 10% of the time, which trains the generic reader used for
   new text.
4. A local convolution (kernel 5) and a one-layer bidirectional Transformer over the
   words (8 heads), adding the right context that a causal backbone lacks.
5. Linear outputs for the seven measures, plus a linear skip from the features of each
   word and its two neighbours on each side.

## Optimisation

| Setting | Value |
|---|---|
| Loss | Huber (δ = 1) on z-scored log durations; binary cross-entropy with soft targets on probabilities |
| Task weights | 1.0 for every measure except `reg_prob` (0.5) |
| Word weights | n / (n + 5) for a word read by n readers; words with fewer than 3 readers are dropped |
| Data mixing | each batch from one dataset, drawn with probability ∝ size^0.5 |
| Batches | up to 16,384 padded tokens and 32 windows |
| Optimizer | AdamW, betas (0.9, 0.98), weight decay 0.01, gradient clipping at 1.0 |
| Learning rate | 1e-3, cosine schedule with 3% warmup |
| Length | up to 20 epochs; validation every 60 steps; stop after 8 evaluations without improvement |
| Model selection | the checkpoint with the best validation macro Pearson r |

## Ensemble members

| Member | Seed | Selected step / steps run | Validation r | Test r | Time on one A40 |
|---|---|---|---|---|---|
| `seed0` | 0 | 5,760 / 6,240 | 0.6485 | 0.6443 | 117 min |
| `seed1` | 1 | 4,620 / 5,100 | 0.6494 | 0.6427 | 95 min |
| `seed2` | 2 | 4,860 / 5,340 | 0.6477 | 0.6432 | 99 min |

- The seeds change the head initialization, dropout and data order; the splits are the
  same.
- Peak GPU memory was 13.7 GB.
- At prediction time the three heads read the same backbone hidden states. Their
  outputs are averaged: durations in log space, so the result is a geometric mean in
  milliseconds, and probabilities as probabilities.

## What was tried and not released

- **Backbone fine-tuning.** The backbone stays frozen (`finetune: frozen`), so the `lora`
  block in the member configs is unused.
- **A per-corpus blend** with the gradient-boosting feature model scores higher on test
  (0.627 vs 0.613 on tier-1 cells), but it needs the feature model at prediction time.
  It is not part of this release.
