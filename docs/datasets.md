# Training data

The model was trained on 31 public reading corpora: 26 eye-tracking (ET) and 5
self-paced reading (SPR) corpora. Together they cover 22 languages, 16 reading tasks,
and native (L1) and second-language (L2) readers. Each corpus was converted to one
row per word of every text. Each row holds the per-word mean over the corpus's readers.
`model_info().datasets` has the same information, plus the reading conditions and reader
groups of every corpus.

- **Readers** is the mean number of readers per word. **Words** counts the word tokens
  of distinct texts. **Measures** lists the targets each corpus provides; a corpus is
  only trained and scored on these.
- Durations come from the corpus's own fixation-level or word-level reports. Durations
  are recomputed with one definition of each measure wherever a corpus ships only
  fixations.
- Words read by fewer than 3 readers are left out. Durations outside 80–6000 ms (ET) or
  100–3000 ms (SPR) are dropped before averaging.
- **Tier 1** (`tier == 1` in `model_info().datasets`) is the subset every baseline was also
  run on: OneStop, InteRead, Provo, GECO, MECO, PoTeC, CopCo and Natural Stories.

**Licences.** No corpus data and no stimulus text is redistributed here: only model
weights and aggregate scores. Every corpus keeps its own licence, so check it before
using the corpora themselves.
- Four corpora are non-commercial: Natural Stories, RaCCooNS and TiNT are
  CC BY-NC-SA 4.0, and InteRead is NC under both of its stated licences. The model
  weights are released under CC BY-NC-SA 4.0 for that reason.
- Corpora marked "not stated" publish no licence; they were used as published for
  research.

<!-- table -->
| Corpus | Paradigm | Languages | Readers (mean) | Words | Measures | Licence |
|---|---|---|---|---|---|---|
| [Brown self-paced reading corpus](https://doi.org/10.1016/j.cognition.2013.02.013) (`brown_spr`) | SPR | en | 19 | 7,188 | rt | not stated |
| [Beijing Sentence Corpus (BSC)](https://osf.io/vr3k8/) (`bsc`) | ET | zh | 54.6 | 1,685 | ffd, gd, gpt, trt, fix_prob, reg_prob | not stated |
| [Beijing Sentence Corpus II (traditional script)](https://osf.io/j9xt4/) (`bsc2`) | ET | zh | 64.5 | 1,685 | ffd, gd, gpt, trt, fix_prob, reg_prob | CC BY 4.0 |
| [CERVINO: eye tracking on Italian legal texts](https://doi.org/10.5281/zenodo.22082820) (`cervino`) | ET | it | 51.2 | 2,593 | ffd, gd, gpt, trt, fix_prob, reg_prob | CC BY 4.0 |
| [Database of Eye-Movement Measures on Words in Chinese Reading](https://osf.io/94wue/) (`chinese_reading`) | ET | zh | 11.7 | 127,914 | ffd, gd, gpt, trt, fix_prob, reg_prob | CC BY 4.0 |
| [Copenhagen Corpus (CopCo)](https://osf.io/ud8s5/) (`copco`) | ET | da | 3.9 | 50,287 | trt | CC BY 4.0 |
| [Eye movements on Machine-generated Texts Corpus (EMTeC)](https://osf.io/ajqze/) (`emtec`) | ET | en | 7.5 | 50,871 | ffd, gd, gpt, trt, fix_prob, reg_prob | CC BY 4.0 |
| [Ghent Eye-tracking Corpus (GECO), English](https://expsy.ugent.be/downloads/geco/) (`geco_en`) | ET | en | 11.6 | 112,816 | ffd, gd, gpt, trt, fix_prob, reg_prob | not stated ("freely available") |
| [Ghent Eye-tracking Corpus (GECO), Dutch](https://expsy.ugent.be/downloads/geco/) (`geco_nl`) | ET | nl | 9 | 60,719 | ffd, gd, gpt, trt, fix_prob, reg_prob | not stated ("freely available") |
| [HeCz: Czech newspaper headlines](http://hdl.handle.net/11234/1-6121) (`hecz`) | SPR | cs | 98.6 | 23,634 | rt | CC BY 4.0 |
| [Potsdam-Allahabad Hindi Eye-tracking Corpus](https://github.com/vasishth/HusainEtAlJEMR2015) (`hindi`) | ET | hi | 29.5 | 2,596 | ffd, gd, gpt, trt, fix_prob | MIT |
| [Hong Kong Corpus of Chinese Sentence and Passage Reading (HKC)](https://osf.io/z465b/) (`hkc`) | ET | zh | 96.9 | 11,797 | ffd, gd, trt, fix_prob, reg_prob | CC BY 4.0 |
| [InteRead: An Eye Tracking Dataset of Interrupted Reading](https://osf.io/43j5f/) (`interead`) | ET | en | 25 | 6,894 | trt | CC BY-NC 4.0 (OSF) / CC BY-NC-SA 4.0 (project page) |
| [Multilingual Eye-movement Corpus (MECO) L1, wave 1](https://osf.io/3527a/) (`meco_l1w1`) | ET | de, el, en, es, et, fi, he, it, ko, nl, no, ru, tr | 33.7 | 25,387 | ffd, gd, gpt, trt, fix_prob, reg_prob | CC BY 4.0 |
| [Multilingual Eye-movement Corpus (MECO) L1, wave 2](https://osf.io/3527a/) (`meco_l1w2`) | ET | da, de, en, es, eu, hi, is, no, pt, ru, sr, tr, zh | 29.7 | 32,903 | ffd, gd, gpt, trt, fix_prob, reg_prob | CC BY 4.0 |
| [Multilingual Eye-movement Corpus (MECO) L2, wave 1](https://osf.io/q9h43/) (`meco_l2w1`) | ET | en | 197.6 | 3,316 | ffd, gd, gpt, trt, fix_prob, reg_prob | CC BY 4.0 |
| [Multilingual Eye-movement Corpus (MECO) L2, wave 2](https://osf.io/q9h43/) (`meco_l2w2`) | ET | en | 234.3 | 3,316 | ffd, gd, gpt, trt, fix_prob, reg_prob | CC BY 4.0 |
| [Natural Stories Corpus](https://github.com/languageMIT/naturalstories) (`natural_stories`) | SPR | en | 82.8 | 10,256 | rt | CC BY-NC-SA 4.0 |
| [OneStop Eye Movements](https://osf.io/2prdq/) (`onestop`) | ET | en | 17.8 | 140,724 | ffd, gd, gpt, trt, fix_prob, reg_prob | CC BY 4.0 |
| [Potsdam Textbook Corpus (PoTeC)](https://osf.io/dn5hp/) (`potec`) | ET | de | 75 | 1,895 | ffd, gd, gpt, trt, fix_prob, reg_prob | CC BY 4.0 (data; the stimulus texts are copyrighted) |
| [The Provo Corpus](https://osf.io/sjefs/) (`provo`) | ET | en | 84 | 2,743 | ffd, gd, gpt, trt, fix_prob, reg_prob | CC BY 4.0 |
| [Persian Sentence Reading Corpus (PSR)](https://osf.io/4w362/) (`psr`) | ET | fa | 59.7 | 1,015 | ffd, gd, gpt, trt, fix_prob, reg_prob | not stated |
| [Radboud Coregistration Corpus of Narrative Sentences (RaCCooNS)](https://doi.org/10.34973/3g69-ha39) (`raccoons`) | ET | nl | 34.4 | 2,783 | ffd, gd, gpt, trt, fix_prob, reg_prob | CC BY-NC-SA 4.0 |
| [RastrOS](https://osf.io/9jxg3/) (`rastros`) | ET | pt | 35.2 | 2,440 | ffd, gd, gpt, trt, fix_prob, reg_prob | CC BY 4.0 |
| [Russian Sentence Corpus (RSC)](https://osf.io/x5q2r/) (`rsc`) | ET | ru | 101.5 | 1,357 | ffd, gd, gpt, trt, fix_prob | CC BY 4.0 |
| [Tilburg corpus of Natural Dutch Texts (TiNT)](https://doi.org/10.34894/0O5XQ7) (`tint`) | SPR | nl | 56.9 | 4,786 | rt | CC BY-NC-SA 4.0 |
| [TURead: An eye movement dataset of Turkish reading](https://osf.io/w53cz/) (`turead`) | ET | tr | 97.4 | 5,886 | ffd, gd, gpt, trt, fix_prob, reg_prob | not stated |
| [UCL corpus, eye tracking](https://doi.org/10.3758/s13428-012-0313-y) (`ucl_et`) | ET | en | 36 | 1,931 | ffd, gd, gpt, trt, fix_prob, reg_prob | not stated |
| [UCL corpus, self-paced reading](https://doi.org/10.3758/s13428-012-0313-y) (`ucl_spr`) | SPR | en | 31.9 | 9,914 | rt | not stated |
| [Zurich Cognitive Language Processing Corpus (ZuCo 1.0), normal reading](https://osf.io/q3zws/) (`zuco1`) | ET | en | 11.7 | 6,588 | ffd, gd, gpt, trt, fix_prob, reg_prob | CC BY 4.0 |
| [Zurich Cognitive Language Processing Corpus (ZuCo 2.0), normal reading](https://osf.io/2urht/) (`zuco2`) | ET | en | 17.6 | 6,828 | ffd, gd, gpt, trt, fix_prob, reg_prob | CC BY 4.0 |

## Citations

- `brown_spr`: Smith, N. J., & Levy, R. (2013). The effect of word predictability on reading time is logarithmic. Cognition, 128(3), 302-319. https://doi.org/10.1016/j.cognition.2013.02.013
- `bsc`: Pan, J., Yan, M., Richter, E. M., Shu, H., & Kliegl, R. (2022). The Beijing Sentence Corpus: A Chinese sentence corpus with eye movement data and predictability norms. Behavior Research Methods, 54(4), 1989-2000. https://doi.org/10.3758/s13428-021-01730-2
- `bsc2`: Yan, M., Pan, J., & Kliegl, R. (2025). The Beijing Sentence Corpus II: A cross-script comparison between traditional and simplified Chinese sentence reading. Behavior Research Methods, 57(2), 60. https://doi.org/10.3758/s13428-024-02523-z
- `cervino`: Scozzaro, C. J., Delsanto, M., Pasquet, S., Revelli, L., Mastropaolo, A., & Radicioni, D. P. (2026). CERVINO: A large-scale eye-tracking resource of Italian legal texts. In Proceedings of EMNLP 2026 (pp. 55269-55283).
- `chinese_reading`: Zhang, G., Yao, P., Ma, G., et al. (2022). The database of eye-movement measures on words in Chinese reading. Scientific Data, 9, 411. https://doi.org/10.1038/s41597-022-01464-6
- `copco`: Hollenstein, N., Barrett, M., & Björnsdóttir, M. (2022). The Copenhagen Corpus of eye tracking recordings from natural reading of Danish texts. In Proceedings of LREC 2022 (pp. 1712-1720). https://aclanthology.org/2022.lrec-1.182
- `emtec`: Bolliger, L. S., Haller, P., Cretton, I. C. R., Reich, D. R., Kew, T., & Jäger, L. A. (2025). EMTeC: A corpus of eye movements on machine-generated texts. Behavior Research Methods, 57(7), 189. https://doi.org/10.3758/s13428-025-02677-4
- `geco_en`: Cop, U., Dirix, N., Drieghe, D., & Duyck, W. (2017). Presenting GECO: An eyetracking corpus of monolingual and bilingual sentence reading. Behavior Research Methods, 49(2), 602-615. https://doi.org/10.3758/s13428-016-0734-0
- `hecz`: Chromý, J., Ceháková, M., & Brand, J. (2025). The HeCz corpus: A large, richly annotated reading corpus of newspaper headlines in Czech. Behavior Research Methods, 57(12), 345. https://doi.org/10.3758/s13428-025-02863-4
- `hindi`: Husain, S., Vasishth, S., & Srinivasan, N. (2015). Integration and prediction difficulty in Hindi sentence comprehension: Evidence from an eye-tracking corpus. Journal of Eye Movement Research, 8(2), 1-12. https://doi.org/10.16910/jemr.8.2.3
- `hkc`: Wu, Y., & Kit, C. (2023). Hong Kong Corpus of Chinese sentence and passage reading. Scientific Data, 10, 899. https://doi.org/10.1038/s41597-023-02813-9
- `interead`: Zermiani, F., Dhar, P., Sood, E., Kögel, F., Bulling, A., & Wirzberger, M. (2024). InteRead: An eye tracking dataset of interrupted reading. In Proceedings of LREC-COLING 2024 (pp. 9154-9169). https://aclanthology.org/2024.lrec-main.802
- `meco_l1w1`: Siegelman, N., Schroeder, S., Acartürk, C., et al. (2022). Expanding horizons of cross-linguistic research on reading: The Multilingual Eye-movement Corpus (MECO). Behavior Research Methods, 54(6), 2843-2863. https://doi.org/10.3758/s13428-021-01772-6
- `meco_l1w2`: Siegelman, N., Schroeder, S., Bao, Y., et al. (2025). Wave 2 of the Multilingual Eye-Movement Corpus (MECO): New text reading data across languages. Scientific Data, 12, 1183. https://doi.org/10.1038/s41597-025-05453-3
- `meco_l2w1`: Kuperman, V., Siegelman, N., Schroeder, S., et al. (2023). Text reading in English as a second language: Evidence from the Multilingual Eye-Movements Corpus. Studies in Second Language Acquisition, 45(1), 3-37. https://doi.org/10.1017/S0272263121000954
- `meco_l2w2`: Kuperman, V., Schroeder, S., Acartürk, C., et al. (2025). New data on text reading in English as a second language: The Wave 2 expansion of the Multilingual Eye-Movement Corpus (MECO). Studies in Second Language Acquisition, 47(2), 677-695. https://doi.org/10.1017/S0272263125000105
- `natural_stories`: Futrell, R., Gibson, E., Tily, H. J., Blank, I., Vishnevetsky, A., Piantadosi, S. T., & Fedorenko, E. (2021). The Natural Stories corpus: A reading-time corpus of English texts containing rare syntactic constructions. Language Resources and Evaluation, 55(1), 63-77. https://doi.org/10.1007/s10579-020-09503-7
- `onestop`: Berzak, Y., Malmaud, J., Shubi, O., Meiri, Y., Lion, E., & Levy, R. (2025). OneStop: A 360-participant English eye tracking dataset with different reading regimes. Scientific Data, 12, 1995. https://doi.org/10.1038/s41597-025-06272-2
- `potec`: Jakobi, D. N., Kern, T., Reich, D. R., Haller, P., & Jäger, L. A. (2025). PoTeC: A German naturalistic eye-tracking-while-reading corpus. Behavior Research Methods, 57(8), 211. https://doi.org/10.3758/s13428-024-02536-8
- `provo`: Luke, S. G., & Christianson, K. (2018). The Provo Corpus: A large eye-tracking corpus with predictability norms. Behavior Research Methods, 50(2), 826-833. https://doi.org/10.3758/s13428-017-0908-4
- `psr`: Soleymani Tekbudak, F., Purmohammad, M., Özkan, A., & Acartürk, C. (2025). The PSR corpus: A Persian sentence reading corpus of eye movements. Behavior Research Methods, 57(1), 14. https://doi.org/10.3758/s13428-024-02517-x
- `raccoons`: Frank, S. L., & Aumeistere, A. (2024). An eye-tracking-with-EEG coregistration corpus of narrative sentences. Language Resources and Evaluation, 58(2), 641-657. https://doi.org/10.1007/s10579-023-09684-x
- `rastros`: Leal, S. E., Lukasova, K., Carthery-Goulart, M. T., & Aluísio, S. M. (2022). RastrOS Project: Natural language processing contributions to the development of an eye-tracking corpus with predictability norms for Brazilian Portuguese. Language Resources and Evaluation, 56(4), 1333-1372. https://doi.org/10.1007/s10579-022-09609-0
- `rsc`: Laurinavichyute, A. K., Sekerina, I. A., Alexeeva, S., Bagdasaryan, K., & Kliegl, R. (2019). Russian Sentence Corpus: Benchmark measures of eye movements in reading in Russian. Behavior Research Methods, 51(3), 1161-1178. https://doi.org/10.3758/s13428-018-1051-6
- `tint`: Østergaard, S. M., Lichtenberg, L. D., Boon, L., & Nicenboim, B. (2026). A corpus of joint EEG and self-paced reading of natural Dutch texts. In Proceedings of LREC 2026 (pp. 11260-11271).
- `turead`: Acartürk, C., Özkan, A., Pekçetin, T. N., Ormanoğlu, Z., & Kırkıcı, B. (2024). TURead: An eye movement dataset of Turkish reading. Behavior Research Methods, 56(3), 1793-1816. https://doi.org/10.3758/s13428-023-02120-6
- `ucl_et`: Frank, S. L., Fernandez Monsalve, I., Thompson, R. L., & Vigliocco, G. (2013). Reading time data for evaluating broad-coverage models of English sentence processing. Behavior Research Methods, 45(4), 1182-1190. https://doi.org/10.3758/s13428-012-0313-y
- `zuco1`: Hollenstein, N., Rotsztejn, J., Troendle, M., Pedroni, A., Zhang, C., & Langer, N. (2018). ZuCo, a simultaneous EEG and eye-tracking resource for natural sentence reading. Scientific Data, 5, 180291. https://doi.org/10.1038/sdata.2018.291
- `zuco2`: Hollenstein, N., Troendle, M., Zhang, C., & Langer, N. (2020). ZuCo 2.0: A dataset of physiological recordings during natural reading and annotation. In Proceedings of LREC 2020 (pp. 138-146). https://aclanthology.org/2020.lrec-1.18
