# LLM baselines

This directory contains the Paper-1 MatchGPT-style open-model prompting baselines:

- Ministral 3 8B
- Phi-4-mini
- Qwen3.5 9B
- Qwen3-1.7B

`test_repaired/` is the canonical repaired untouched-test evaluation: 406,908 unique generated and evaluated rows, complete across all 24 model-by-dataset cells. It includes pooled results, seed-level results, full predictions, publication tables, and the repair manifest.

The validation CSV files preserve prompt selection across direct/rationale prompting and 0–4 demonstrations. The LLM results accompany Experiment 5 in the consolidated paper bundle, but they are not part of the 13-family, 5,616-cell supervised label-efficiency matrix.
