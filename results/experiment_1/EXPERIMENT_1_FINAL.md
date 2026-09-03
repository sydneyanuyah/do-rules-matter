# Experiment 1 — Classification

Status: **COMPLETE**.

This canonical package contains 47 classification configurations evaluated on six locked public datasets. It supersedes the earlier 39/40-row consolidations. Experiment 2 ranking outputs are intentionally excluded.

## Coverage

- Configurations: 47
- Datasets: 6
- Completed model–dataset cells: 282/282 (100%)
- Seeds for stochastic systems: 20260725, 20260726, 20260727
- Standard deviation: sample SD across the three protocol seeds; blank only for deterministic single-run systems
- Public genealogy benchmark: Link-Lives Release 2
- Private data: excluded

## Included model families

- deterministic: 1
- frozen_embedding: 7
- scalar_fusion: 2
- hef_linear: 7
- hef_gbdt: 7
- neural_em: 7
- mixda_hef: 4
- cross_evidence: 3
- joint_neural_hef: 7
- neural_hef_oof: 2

## Best held-out F1 by dataset

| Dataset | Model | Mean F1 | SD |
|---|---|---:|---:|
| Abt-Buy | Tuned RoBERTa | 0.9081 | 0.0064 |
| Amazon-Google | Joint neural HEF + JINA (fine-tuned) | 0.7864 | 0.0216 |
| Walmart-Amazon | HEF-GBDT + E5 + fine-tuned RoBERTa (OOF) | 0.8864 | 0.0028 |
| WDC seen | HEF-GBDT + E5 + fine-tuned Jina (OOF) | 0.7577 | 0.0065 |
| WDC unseen | HEF-GBDT + E5 + fine-tuned Jina (OOF) | 0.7496 | 0.0085 |
| Link-Lives | Official Ditto + MixDA | 0.9453 | 0.0011 |

## Protocol

All stochastic systems use the fixed protocol seeds. Model and threshold selection use validation data only; reported means are from untouched test evaluation. Task-trained neural evidence used in HEF hybrids is record-grouped out-of-fold evidence, preventing in-sample stacking leakage.

## Files

- `Experiment_1_Classification_Results.xlsx`: formatted canonical table, scope, and provenance
- `experiment_1_results.csv`: one row per configuration
- `experiment_1_results_long.csv`: one row per configuration–dataset cell
- `experiment_1_completeness.json`: machine-readable completion declaration
- `manifest.json`: file hashes and package inventory
