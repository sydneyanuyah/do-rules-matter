# Experiment 5 — Final Consolidated Results

Status: **complete**. The strict matrix contains 13 model families × 6 datasets × 24 label fractions × 3 fixed seeds = **5,616 verified cells**.

Means and sample standard deviations are reported from the three fixed seeds. Model selection used validation data only; test evaluation remained untouched; task-trained stacking inputs use leakage-safe out-of-fold predictions.

The `LLM_Baselines/` directory contains the four open-model prompted baselines (Ministral 3 8B, Phi-4-mini, Qwen3.5 9B, and Qwen3-1.7B), including validation audits and the repaired 406,908-row untouched-test evaluation. These LLM results are preserved alongside Experiment 5 for the paper bundle but are not counted as part of the 5,616 supervised label-efficiency cells.
