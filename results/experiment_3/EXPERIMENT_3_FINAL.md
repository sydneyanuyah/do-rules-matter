# Experiment 3 — Expanded Fusion and Model Comparison

Status: **COMPLETE**.

This package supersedes the earlier 39-configuration consolidation. It contains 47 configurations and explicitly includes seven jointly fine-tuned neural HEF systems plus two leakage-safe OOF neural-evidence HEF systems.

## Neural HEF coverage

- Joint neural HEF classification: 126/126 seed cells
- Joint neural HEF ranking: 126/126 seed cells
- OOF neural-evidence HEF classification: 36/36 seed cells
- OOF neural-evidence HEF ranking: 36/36 seed cells
- Missing applicable primary cells: 0

## Neural HEF results

Unweighted macro mean across the six locked datasets:

| Configuration | Classification F1 | Ranking MRR@100 |
|---|---:|---:|
| HEF-GBDT + E5 + fine-tuned RoBERTa (OOF) | 0.8172 | 0.9161 |
| HEF-GBDT + E5 + fine-tuned Jina (OOF) | 0.8279 | 0.9154 |
| Joint neural HEF + RoBERTa (fine-tuned) | 0.8194 | 0.8852 |
| Joint neural HEF + Jina (fine-tuned) | 0.8292 | 0.8770 |
| Joint neural HEF + E5 (fine-tuned) | 0.7408 | 0.8417 |
| Joint neural HEF + BERT (fine-tuned) | 0.7760 | 0.8362 |
| Joint neural HEF + BGE (fine-tuned) | 0.7189 | 0.8310 |
| Joint neural HEF + MiniLM (fine-tuned) | 0.6913 | 0.8264 |
| Joint neural HEF + GTE (fine-tuned) | 0.7820 | 0.8087 |

## Protocol

The expanded consolidation reuses completed, validation-selected test evaluations from canonical Experiment 1 classification and Experiment 2 ranking artifacts. It does not refit on test data. Task-trained neural evidence supplied to a HEF meta-learner is generated out of fold with zero group overlap.
