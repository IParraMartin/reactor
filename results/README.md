# Results

Evaluation tables of the released model (`reactor-qwen3.5-4b`, v0.1.0), written by the
training code. [docs/evaluation.md](../docs/evaluation.md) explains the metrics.

| File | Contents |
|---|---|
| `summary.json` | Test and validation macro scores, per-measure means and the ensemble members |
| `metrics_test.csv`, `metrics_val.csv` | Every scored cell (dataset, condition, reader group, language, measure): Pearson, Spearman, R², MAE, words, noise ceiling, r / ceiling |
| `per_dataset.csv`, `per_language.csv`, `per_measure.csv` | Test means by dataset, language and measure |
| `benchmark.csv` | Every system on the 239 tier-1 test cells |
| `comparisons.csv` | Paired bootstrap differences (reactor minus the other system), overall, per measure and per dataset |
| `heldout.csv`, `heldout_comparison.csv` | The held-out-corpus experiment (Provo, PoTeC and CopCo left out of training) |
| `datasets.csv` | Size and measure coverage of every training corpus |
| `training/` | Every ensemble member's configuration, training log and selected step (`members.csv`) |

MAE is in milliseconds for durations and in probability units for `fix_prob` and
`reg_prob`. In `comparisons.csv` and `heldout_comparison.csv`, `run` is the first system
of each comparison and `difference` is `pearson_run - pearson_other`.
