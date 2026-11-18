# Notices

## Code

The `reactor` package is released under the MIT licence ([LICENSE](LICENSE)).

## Model weights

The weights of the heads on the Hugging Face Hub
([IParraMartin/reactor-qwen3.5-4b](https://huggingface.co/IParraMartin/reactor-qwen3.5-4b))
are released under the
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/)
licence.

## Backbone

The heads read the hidden states of
[Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base) (Copyright 2026 Alibaba
Cloud, Apache License 2.0). It is not redistributed: it is downloaded from its own
repository, under its own licence, when a model is loaded.

## Training corpora

The heads were trained on word-level averages of 31 reading corpora. No corpus data,
participant data or stimulus text is distributed with this package or the model; the
`results/` tables contain only aggregate scores. Every corpus keeps its own licence:

| Licence | Corpora |
|---|---|
| CC BY 4.0 | OneStop, Provo, MECO L1 waves 1 and 2, MECO L2 waves 1 and 2, PoTeC (data), CopCo, EMTeC, ZuCo 1.0, ZuCo 2.0, RastrOS, CERVINO, RSC, BSC II, Chinese Reading database, HKC, HeCz |
| CC BY-NC-SA 4.0 | Natural Stories, RaCCooNS, TiNT |
| CC BY-NC 4.0 (OSF) / CC BY-NC-SA 4.0 (project page) | InteRead |
| MIT (Copyright 2017 Shravan Vasishth) | Potsdam-Allahabad Hindi Eye-tracking Corpus |
| Not stated | GECO (English, Dutch), UCL (eye tracking, self-paced reading), BSC, TURead, PSR, Brown |

**Caveats.**
- The stimulus texts of PoTeC are copyrighted and licensed to its authors only.
- Restricted TiNT files are available for non-commercial academic use only, with no
  redistribution.
- Corpora with no stated licence were used as published, for research.

Sources and citations are in [docs/datasets.md](docs/datasets.md).
