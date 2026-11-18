"""Builds the packaged model description and the dataset documentation of a release.

Reads the released model's ``config.json`` and the result tables in ``results/`` (written
by the training code's ``scripts/model_card_results.py``), and writes
``src/reactor/resources/model_info.json``, which ``reactor.model_info()`` serves, and the
dataset table of ``docs/datasets.md``. The dataset sources, licences and citations below
were checked against each corpus's official release page.

Example: python scripts/build_model_info.py out/reactor-qwen3.5-4b/config.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from reactor.hub import DEFAULT_REPO, DEFAULT_REVISION
from reactor.measures import DESCRIPTIONS
from reactor.text import LANGUAGE_NAMES

ROOT = Path(__file__).resolve().parents[1]
MEASURE_ORDER = ["ffd", "gd", "gpt", "trt", "fix_prob", "reg_prob", "rt"]

SOURCES = {
    "onestop": {
        "full_name": "OneStop Eye Movements",
        "licence": "CC BY 4.0",
        "url": "https://osf.io/2prdq/",
        "citation": "Berzak, Y., Malmaud, J., Shubi, O., Meiri, Y., Lion, E., & Levy, R. (2025). OneStop: A 360-participant English eye tracking dataset with different reading regimes. Scientific Data, 12, 1995. https://doi.org/10.1038/s41597-025-06272-2"
    },
    "interead": {
        "full_name": "InteRead: An Eye Tracking Dataset of Interrupted Reading",
        "licence": "CC BY-NC 4.0 (OSF) / CC BY-NC-SA 4.0 (project page)",
        "url": "https://osf.io/43j5f/",
        "citation": "Zermiani, F., Dhar, P., Sood, E., Kögel, F., Bulling, A., & Wirzberger, M. (2024). InteRead: An eye tracking dataset of interrupted reading. In Proceedings of LREC-COLING 2024 (pp. 9154-9169). https://aclanthology.org/2024.lrec-main.802"
    },
    "provo": {
        "full_name": "The Provo Corpus",
        "licence": "CC BY 4.0",
        "url": "https://osf.io/sjefs/",
        "citation": "Luke, S. G., & Christianson, K. (2018). The Provo Corpus: A large eye-tracking corpus with predictability norms. Behavior Research Methods, 50(2), 826-833. https://doi.org/10.3758/s13428-017-0908-4"
    },
    "geco_en": {
        "full_name": "Ghent Eye-tracking Corpus (GECO), English",
        "licence": "not stated (\"freely available\")",
        "url": "https://expsy.ugent.be/downloads/geco/",
        "citation": "Cop, U., Dirix, N., Drieghe, D., & Duyck, W. (2017). Presenting GECO: An eyetracking corpus of monolingual and bilingual sentence reading. Behavior Research Methods, 49(2), 602-615. https://doi.org/10.3758/s13428-016-0734-0"
    },
    "geco_nl": {
        "full_name": "Ghent Eye-tracking Corpus (GECO), Dutch",
        "licence": "not stated (\"freely available\")",
        "url": "https://expsy.ugent.be/downloads/geco/",
        "citation": "Cop, U., Dirix, N., Drieghe, D., & Duyck, W. (2017). Presenting GECO: An eyetracking corpus of monolingual and bilingual sentence reading. Behavior Research Methods, 49(2), 602-615. https://doi.org/10.3758/s13428-016-0734-0"
    },
    "meco_l1w1": {
        "full_name": "Multilingual Eye-movement Corpus (MECO) L1, wave 1",
        "licence": "CC BY 4.0",
        "url": "https://osf.io/3527a/",
        "citation": "Siegelman, N., Schroeder, S., Acartürk, C., et al. (2022). Expanding horizons of cross-linguistic research on reading: The Multilingual Eye-movement Corpus (MECO). Behavior Research Methods, 54(6), 2843-2863. https://doi.org/10.3758/s13428-021-01772-6"
    },
    "meco_l1w2": {
        "full_name": "Multilingual Eye-movement Corpus (MECO) L1, wave 2",
        "licence": "CC BY 4.0",
        "url": "https://osf.io/3527a/",
        "citation": "Siegelman, N., Schroeder, S., Bao, Y., et al. (2025). Wave 2 of the Multilingual Eye-Movement Corpus (MECO): New text reading data across languages. Scientific Data, 12, 1183. https://doi.org/10.1038/s41597-025-05453-3"
    },
    "meco_l2w1": {
        "full_name": "Multilingual Eye-movement Corpus (MECO) L2, wave 1",
        "licence": "CC BY 4.0",
        "url": "https://osf.io/q9h43/",
        "citation": "Kuperman, V., Siegelman, N., Schroeder, S., et al. (2023). Text reading in English as a second language: Evidence from the Multilingual Eye-Movements Corpus. Studies in Second Language Acquisition, 45(1), 3-37. https://doi.org/10.1017/S0272263121000954"
    },
    "meco_l2w2": {
        "full_name": "Multilingual Eye-movement Corpus (MECO) L2, wave 2",
        "licence": "CC BY 4.0",
        "url": "https://osf.io/q9h43/",
        "citation": "Kuperman, V., Schroeder, S., Acartürk, C., et al. (2025). New data on text reading in English as a second language: The Wave 2 expansion of the Multilingual Eye-Movement Corpus (MECO). Studies in Second Language Acquisition, 47(2), 677-695. https://doi.org/10.1017/S0272263125000105"
    },
    "potec": {
        "full_name": "Potsdam Textbook Corpus (PoTeC)",
        "licence": "CC BY 4.0 (data; the stimulus texts are copyrighted)",
        "url": "https://osf.io/dn5hp/",
        "citation": "Jakobi, D. N., Kern, T., Reich, D. R., Haller, P., & Jäger, L. A. (2025). PoTeC: A German naturalistic eye-tracking-while-reading corpus. Behavior Research Methods, 57(8), 211. https://doi.org/10.3758/s13428-024-02536-8"
    },
    "copco": {
        "full_name": "Copenhagen Corpus (CopCo)",
        "licence": "CC BY 4.0",
        "url": "https://osf.io/ud8s5/",
        "citation": "Hollenstein, N., Barrett, M., & Björnsdóttir, M. (2022). The Copenhagen Corpus of eye tracking recordings from natural reading of Danish texts. In Proceedings of LREC 2022 (pp. 1712-1720). https://aclanthology.org/2022.lrec-1.182"
    },
    "natural_stories": {
        "full_name": "Natural Stories Corpus",
        "licence": "CC BY-NC-SA 4.0",
        "url": "https://github.com/languageMIT/naturalstories",
        "citation": "Futrell, R., Gibson, E., Tily, H. J., Blank, I., Vishnevetsky, A., Piantadosi, S. T., & Fedorenko, E. (2021). The Natural Stories corpus: A reading-time corpus of English texts containing rare syntactic constructions. Language Resources and Evaluation, 55(1), 63-77. https://doi.org/10.1007/s10579-020-09503-7"
    },
    "emtec": {
        "full_name": "Eye movements on Machine-generated Texts Corpus (EMTeC)",
        "licence": "CC BY 4.0",
        "url": "https://osf.io/ajqze/",
        "citation": "Bolliger, L. S., Haller, P., Cretton, I. C. R., Reich, D. R., Kew, T., & Jäger, L. A. (2025). EMTeC: A corpus of eye movements on machine-generated texts. Behavior Research Methods, 57(7), 189. https://doi.org/10.3758/s13428-025-02677-4"
    },
    "ucl_et": {
        "full_name": "UCL corpus, eye tracking",
        "licence": "not stated",
        "url": "https://doi.org/10.3758/s13428-012-0313-y",
        "citation": "Frank, S. L., Fernandez Monsalve, I., Thompson, R. L., & Vigliocco, G. (2013). Reading time data for evaluating broad-coverage models of English sentence processing. Behavior Research Methods, 45(4), 1182-1190. https://doi.org/10.3758/s13428-012-0313-y"
    },
    "ucl_spr": {
        "full_name": "UCL corpus, self-paced reading",
        "licence": "not stated",
        "url": "https://doi.org/10.3758/s13428-012-0313-y",
        "citation": "Frank, S. L., Fernandez Monsalve, I., Thompson, R. L., & Vigliocco, G. (2013). Reading time data for evaluating broad-coverage models of English sentence processing. Behavior Research Methods, 45(4), 1182-1190. https://doi.org/10.3758/s13428-012-0313-y"
    },
    "zuco1": {
        "full_name": "Zurich Cognitive Language Processing Corpus (ZuCo 1.0), normal reading",
        "licence": "CC BY 4.0",
        "url": "https://osf.io/q3zws/",
        "citation": "Hollenstein, N., Rotsztejn, J., Troendle, M., Pedroni, A., Zhang, C., & Langer, N. (2018). ZuCo, a simultaneous EEG and eye-tracking resource for natural sentence reading. Scientific Data, 5, 180291. https://doi.org/10.1038/sdata.2018.291"
    },
    "zuco2": {
        "full_name": "Zurich Cognitive Language Processing Corpus (ZuCo 2.0), normal reading",
        "licence": "CC BY 4.0",
        "url": "https://osf.io/2urht/",
        "citation": "Hollenstein, N., Troendle, M., Zhang, C., & Langer, N. (2020). ZuCo 2.0: A dataset of physiological recordings during natural reading and annotation. In Proceedings of LREC 2020 (pp. 138-146). https://aclanthology.org/2020.lrec-1.18"
    },
    "raccoons": {
        "full_name": "Radboud Coregistration Corpus of Narrative Sentences (RaCCooNS)",
        "licence": "CC BY-NC-SA 4.0",
        "url": "https://doi.org/10.34973/3g69-ha39",
        "citation": "Frank, S. L., & Aumeistere, A. (2024). An eye-tracking-with-EEG coregistration corpus of narrative sentences. Language Resources and Evaluation, 58(2), 641-657. https://doi.org/10.1007/s10579-023-09684-x"
    },
    "rastros": {
        "full_name": "RastrOS",
        "licence": "CC BY 4.0",
        "url": "https://osf.io/9jxg3/",
        "citation": "Leal, S. E., Lukasova, K., Carthery-Goulart, M. T., & Aluísio, S. M. (2022). RastrOS Project: Natural language processing contributions to the development of an eye-tracking corpus with predictability norms for Brazilian Portuguese. Language Resources and Evaluation, 56(4), 1333-1372. https://doi.org/10.1007/s10579-022-09609-0"
    },
    "cervino": {
        "full_name": "CERVINO: eye tracking on Italian legal texts",
        "licence": "CC BY 4.0",
        "url": "https://doi.org/10.5281/zenodo.22082820",
        "citation": "Scozzaro, C. J., Delsanto, M., Pasquet, S., Revelli, L., Mastropaolo, A., & Radicioni, D. P. (2026). CERVINO: A large-scale eye-tracking resource of Italian legal texts. In Proceedings of EMNLP 2026 (pp. 55269-55283)."
    },
    "rsc": {
        "full_name": "Russian Sentence Corpus (RSC)",
        "licence": "CC BY 4.0",
        "url": "https://osf.io/x5q2r/",
        "citation": "Laurinavichyute, A. K., Sekerina, I. A., Alexeeva, S., Bagdasaryan, K., & Kliegl, R. (2019). Russian Sentence Corpus: Benchmark measures of eye movements in reading in Russian. Behavior Research Methods, 51(3), 1161-1178. https://doi.org/10.3758/s13428-018-1051-6"
    },
    "hindi": {
        "full_name": "Potsdam-Allahabad Hindi Eye-tracking Corpus",
        "licence": "MIT",
        "url": "https://github.com/vasishth/HusainEtAlJEMR2015",
        "citation": "Husain, S., Vasishth, S., & Srinivasan, N. (2015). Integration and prediction difficulty in Hindi sentence comprehension: Evidence from an eye-tracking corpus. Journal of Eye Movement Research, 8(2), 1-12. https://doi.org/10.16910/jemr.8.2.3"
    },
    "bsc": {
        "full_name": "Beijing Sentence Corpus (BSC)",
        "licence": "not stated",
        "url": "https://osf.io/vr3k8/",
        "citation": "Pan, J., Yan, M., Richter, E. M., Shu, H., & Kliegl, R. (2022). The Beijing Sentence Corpus: A Chinese sentence corpus with eye movement data and predictability norms. Behavior Research Methods, 54(4), 1989-2000. https://doi.org/10.3758/s13428-021-01730-2"
    },
    "bsc2": {
        "full_name": "Beijing Sentence Corpus II (traditional script)",
        "licence": "CC BY 4.0",
        "url": "https://osf.io/j9xt4/",
        "citation": "Yan, M., Pan, J., & Kliegl, R. (2025). The Beijing Sentence Corpus II: A cross-script comparison between traditional and simplified Chinese sentence reading. Behavior Research Methods, 57(2), 60. https://doi.org/10.3758/s13428-024-02523-z"
    },
    "chinese_reading": {
        "full_name": "Database of Eye-Movement Measures on Words in Chinese Reading",
        "licence": "CC BY 4.0",
        "url": "https://osf.io/94wue/",
        "citation": "Zhang, G., Yao, P., Ma, G., et al. (2022). The database of eye-movement measures on words in Chinese reading. Scientific Data, 9, 411. https://doi.org/10.1038/s41597-022-01464-6"
    },
    "hkc": {
        "full_name": "Hong Kong Corpus of Chinese Sentence and Passage Reading (HKC)",
        "licence": "CC BY 4.0",
        "url": "https://osf.io/z465b/",
        "citation": "Wu, Y., & Kit, C. (2023). Hong Kong Corpus of Chinese sentence and passage reading. Scientific Data, 10, 899. https://doi.org/10.1038/s41597-023-02813-9"
    },
    "turead": {
        "full_name": "TURead: An eye movement dataset of Turkish reading",
        "licence": "not stated",
        "url": "https://osf.io/w53cz/",
        "citation": "Acartürk, C., Özkan, A., Pekçetin, T. N., Ormanoğlu, Z., & Kırkıcı, B. (2024). TURead: An eye movement dataset of Turkish reading. Behavior Research Methods, 56(3), 1793-1816. https://doi.org/10.3758/s13428-023-02120-6"
    },
    "psr": {
        "full_name": "Persian Sentence Reading Corpus (PSR)",
        "licence": "not stated",
        "url": "https://osf.io/4w362/",
        "citation": "Soleymani Tekbudak, F., Purmohammad, M., Özkan, A., & Acartürk, C. (2025). The PSR corpus: A Persian sentence reading corpus of eye movements. Behavior Research Methods, 57(1), 14. https://doi.org/10.3758/s13428-024-02517-x"
    },
    "brown_spr": {
        "full_name": "Brown self-paced reading corpus",
        "licence": "not stated",
        "url": "https://doi.org/10.1016/j.cognition.2013.02.013",
        "citation": "Smith, N. J., & Levy, R. (2013). The effect of word predictability on reading time is logarithmic. Cognition, 128(3), 302-319. https://doi.org/10.1016/j.cognition.2013.02.013"
    },
    "tint": {
        "full_name": "Tilburg corpus of Natural Dutch Texts (TiNT)",
        "licence": "CC BY-NC-SA 4.0",
        "url": "https://doi.org/10.34894/0O5XQ7",
        "citation": "Østergaard, S. M., Lichtenberg, L. D., Boon, L., & Nicenboim, B. (2026). A corpus of joint EEG and self-paced reading of natural Dutch texts. In Proceedings of LREC 2026 (pp. 11260-11271)."
    },
    "hecz": {
        "full_name": "HeCz: Czech newspaper headlines",
        "licence": "CC BY 4.0",
        "url": "http://hdl.handle.net/11234/1-6121",
        "citation": "Chromý, J., Ceháková, M., & Brand, J. (2025). The HeCz corpus: A large, richly annotated reading corpus of newspaper headlines in Czech. Behavior Research Methods, 57(12), 345. https://doi.org/10.3758/s13428-025-02863-4"
    }
}
"""Full name, licence, release page and primary citation of every training corpus."""

TIER1 = (
    "onestop",
    "interead",
    "provo",
    "geco_en",
    "geco_nl",
    "meco_l1w1",
    "meco_l1w2",
    "meco_l2w1",
    "meco_l2w2",
    "potec",
    "copco",
    "natural_stories"
)
"""The first twelve corpora, on which every baseline was also run."""


def records(table: pd.DataFrame, digits: int = 4) -> list[dict]:
    """A table as JSON records, with floats rounded."""
    return json.loads(table.round(digits).to_json(orient="records", force_ascii=False))


def dataset_entries(config: dict, results: Path) -> list[dict]:
    """Describes every training corpus: source, licence, size and reading setups."""
    coverage = pd.read_csv(results / "datasets.csv").set_index("dataset")
    metrics = pd.read_csv(results / "metrics_test.csv")
    domains: dict[str, list[tuple[str, str]]] = {}
    for key in config["stats"]["domain"]:
        dataset, condition, reader_group = key.split("|")
        domains.setdefault(dataset, []).append((condition, reader_group))
    entries = []
    for name in sorted(SOURCES):
        row = coverage.loc[name]
        measures = [m for m in MEASURE_ORDER if row.get(f"{m}_coverage", 0) > 0]
        entries.append({
            "name": name,
            **SOURCES[name],
            "paradigm": "spr" if measures == ["rt"] else "et",
            "languages": row["languages"].split(","),
            "conditions": sorted({c for c, _ in domains.get(name, [])}),
            "reader_groups": sorted({g for _, g in domains.get(name, [])}),
            "measures": measures,
            "texts": int(row["texts"]),
            "words": int(row["words"]),
            "mean_readers": round(float(row["mean_readers"]), 1),
            "tier": 1 if name in TIER1 else 2,
            "test_cells": int((metrics["dataset"] == name).sum())
        })
    return entries


def training_entry(config: dict, results: Path) -> dict:
    """Summarizes how the released heads were trained."""
    train_cfg = json.loads((results / "training" / "seed0_config.json").read_text(encoding="utf-8"))
    data, train = train_cfg["data"], train_cfg["train"]
    members = pd.read_csv(results / "training" / "members.csv")
    logs = [pd.read_csv(results / "training" / f"{m}_train_log.csv") for m in members["member"]]
    return {
        "summary": (
            "Heads trained on a frozen Qwen3.5-4B-Base with 31 corpora mixed by temperature "
            "sampling; three seeds, each the checkpoint with the best validation score, "
            "averaged at prediction time."
        ),
        "data": {
            "datasets": len(data["datasets"]),
            "split": "80/10/10 by text, within each dataset; translations (MECO) and text versions (OneStop) share a split",
            "split_seed": data["split_seed"],
            "min_readers": data["min_readers"],
            "reader_weight": f"n / (n + {data['reliability_k']:g}) for a word read by n readers",
            "sampling": f"dataset drawn with probability proportional to size^{train['alpha']}",
            "duration_bounds_ms": {"et": [80, 6000], "spr": [100, 3000]}
        },
        "targets": {
            "measures": {m: DESCRIPTIONS[m] for m in MEASURE_ORDER},
            "aggregation": "durations: mean log duration over readers (eye tracking: readers who fixated the word); probabilities: share of readers",
            "normalization": "log durations z-scored per dataset, condition and reader group"
        },
        "inputs": {
            "prompt": "Language: {language}. Measure: {paradigm}. Task: {condition}. Readers: {native|non-native} speakers.",
            "features": ["log_length", "zipf", "zipf_missing", "surprisal", "entropy", "relative_position", "sentence_start", "sentence_end", "punctuation_end", "line_start", "line_end", "line_known"],
            "windows": config["windows"],
            "pool": f"hidden state of each word's {config['pool']} subword token"
        },
        "architecture": {
            "backbone": f"{config['backbone']['name']} (frozen, {config['backbone']['dtype']})",
            "head": config["head"],
            "description": "two learned layer mixes, feature fusion, dataset FiLM (dropped 10% of the time to learn a generic reader), local convolution plus a one-layer bidirectional word Transformer, linear outputs with a feature skip"
        },
        "loss": {
            "durations": f"Huber (delta {train['huber_delta']}) on z-scored log durations",
            "probabilities": "binary cross-entropy with soft targets",
            "task_weights": train["task_weights"],
            "masking": "measures a corpus does not provide are left out of the loss"
        },
        "optimisation": {
            "optimizer": "AdamW (betas 0.9, 0.98)",
            "learning_rate": train["lr_head"],
            "schedule": f"cosine with {train['warmup_ratio']:.0%} warmup",
            "weight_decay": train["weight_decay"],
            "max_grad_norm": train["max_grad_norm"],
            "batch": f"up to {train['max_tokens_per_batch']} padded tokens and {train['max_batch_size']} windows",
            "max_epochs": train["epochs"],
            "early_stopping": f"validation macro Pearson every {train['eval_every']} steps, patience {train['patience']}"
        },
        "compute": {
            "hardware": "one NVIDIA A40 (48 GB) per seed",
            "minutes_per_seed": [round(float(log["elapsed_min"].iloc[-1])) for log in logs],
            "peak_memory_gb": round(max(float(log["peak_memory_gb"].max()) for log in logs), 1)
        },
        "members": records(members)
    }


def results_entry(results: Path) -> dict:
    """The evaluation summaries served by ``model_info().results``."""
    summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    comparisons = pd.read_csv(results / "comparisons.csv")
    keep = ~comparisons["scope"].str.startswith("dataset:")
    columns = ["comparison", "scope", "cells", "pearson_run", "pearson_other", "difference", "ci_low", "ci_high"]
    return {
        "metric": "Pearson r between predicted and observed word means, per dataset x condition x reader group x language x measure cell, averaged over cells (cell_macro) or first within each dataset (dataset_macro)",
        "test": {k: round(v, 4) for k, v in summary["test"].items()},
        "val": {k: round(v, 4) for k, v in summary["val"].items()},
        "per_measure": records(pd.read_csv(results / "per_measure.csv")),
        "per_dataset": records(pd.read_csv(results / "per_dataset.csv")),
        "per_language": records(pd.read_csv(results / "per_language.csv").drop(columns="datasets")),
        "benchmark": records(pd.read_csv(results / "benchmark.csv").drop(columns="run")),
        "comparisons": records(comparisons.loc[keep, columns]),
        "heldout": records(pd.read_csv(results / "heldout.csv"))
    }


def datasets_markdown(entries: list[dict]) -> str:
    """The dataset table of ``docs/datasets.md``."""
    lines = [
        "| Corpus | Paradigm | Languages | Readers (mean) | Words | Measures | Licence |",
        "|---|---|---|---|---|---|---|"
    ]
    for e in entries:
        lines.append(
            f"| [{e['full_name']}]({e['url']}) (`{e['name']}`) | {e['paradigm'].upper()} | {', '.join(e['languages'])} "
            f"| {e['mean_readers']:g} | {e['words']:,} | {', '.join(e['measures'])} | {e['licence']} |"
        )
    lines += ["", "## Citations", ""]
    seen = set()
    for e in entries:
        if e["citation"] not in seen:
            seen.add(e["citation"])
            lines.append(f"- `{e['name']}`: {e['citation']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="config.json of the released model")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    entries = dataset_entries(config, args.results)
    languages = sorted({key.split("|")[0] for key in config["stats"]["language"]})
    info = {
        "name": DEFAULT_REPO,
        "revision": DEFAULT_REVISION,
        "licence": "CC-BY-NC-SA-4.0",
        "backbone": config["backbone"],
        "training": training_entry(config, args.results),
        "datasets": entries,
        "languages": {code: LANGUAGE_NAMES[code] for code in languages},
        "conditions": sorted({key.split("|")[1] for key in config["stats"]["domain"]}),
        "results": results_entry(args.results)
    }
    out = ROOT / "src" / "reactor" / "resources" / "model_info.json"
    out.write_text(json.dumps(info, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    docs = ROOT / "docs" / "datasets.md"
    text = docs.read_text(encoding="utf-8") if docs.exists() else "# Training data\n\n<!-- table -->\n"
    head = text.split("<!-- table -->")[0]
    docs.write_text(head + "<!-- table -->\n" + datasets_markdown(entries), encoding="utf-8")
    print(f"{out}: {len(entries)} datasets, {len(languages)} languages")


if __name__ == "__main__":
    main()
