# Public result bundle

This directory contains compact, publication-facing outputs for the six public datasets only.

| Directory | Contents |
|---|---|
| `experiment_1` | Classification matrix, per-dataset means/SDs, completeness checks, and workbook |
| `experiment_2` | Candidate-ranking run table, summary, and manifest |
| `experiment_3` | Full configuration registry, classification/ranking summaries, per-run metrics, and workbook |
| `experiment_4` | Backbone-transfer classification and ranking summaries |
| `experiment_5` | Label-efficiency summaries across fractions and seeds, plus compact LLM evaluation summaries |
| `experiment_6` | Evidence-ablation metrics and manifests |
| `experiment_7` | Controlled evidence-loss metrics and manifests |
| `experiment_8` | Efficiency/deployment measurements and summaries |
| `experiment_9` | Public-dataset bootstrap, effect-size, raw-p, and Holm-adjusted statistical tables |

Large score arrays, raw predictions, model weights, caches, and dataset files are intentionally excluded. Values marked missing in a source table remain missing; this release does not synthesize results. Relative provenance paths refer to the artifact layout created by the runners.
