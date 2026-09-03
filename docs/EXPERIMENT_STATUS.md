# Public experiment release status

This release contains only the six public benchmarks and Experiments 1–9. The compact tables in `results/` are the publication-facing snapshot; large intermediate artifacts are excluded from Git.

| Experiment | Scope represented in this release |
|---|---|
| 1 — Classification | Rules, frozen embeddings, HEF-Linear, HEF-GBDT, tuned neural comparators, official Ditto/MixDA, AnyMatch, OOF hybrids, and joint neural HEF variants across six public datasets and fixed seeds |
| 2 — Candidate ranking | Standalone and fusion rerankers on fixed label-blind top-100 pools, with conditional ranking metrics and end-to-end coverage/Hits@100 |
| 3 — Model/fusion comparison | Consolidated configuration registry and classification/ranking comparison tables for the available public systems |
| 4 — Backbone transfer | Classification and ranking transfer across the configured frozen semantic backbones |
| 5 — Label efficiency | Requested 24 label fractions and three fixed seeds for the expanded public model families; compact aggregate tables are included |
| 6 — Evidence ablation | Removal of lexical, numerical, categorical, relational, rule, availability, and semantic evidence, plus raw-field and aggregate-rule controls |
| 7 — Robustness | Controlled evidence masking at 0%, 10%, 30%, 50%, and 70% for the selected public rerankers |
| 8 — Efficiency/deployment | Cold end-to-end and warm/cached latency, throughput, memory, model-size, and cost measurements |
| 9 — Statistical validation | Paired bootstrap confidence intervals, effect sizes, raw p-values, and Holm-adjusted comparisons on public datasets |

Completion must be interpreted from each experiment's manifest and completeness table. A model is not considered repeated merely because a deterministic baseline has one identical value; learned systems report the fixed three-seed protocol and sample standard deviation where applicable.
