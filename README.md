# Hybrid Evidence Fusion for Entity Matching

This repository contains the reproducible research code and compact result tables for **Hybrid Evidence Fusion (HEF)**, a public-benchmark entity-matching study. HEF combines structured evidence, rules, semantic representations, and learned fusion models for pair classification and candidate ranking.

The release covers six public datasets: WDC Products (seen and unseen), Abt–Buy, Amazon–Google, Walmart–Amazon, and Link-Lives Release 2.

## What is included

- A Python package for validation, serialization, feature construction, embedding, fusion, ranking, and evaluation.
- Locked experiment configuration, model revisions, seeds, metrics, and data policies.
- Experiment runners for classification, ranking, label efficiency, ablation, controlled evidence loss, efficiency, and statistical analysis (Experiments 1–9).
- Expanded leakage-safe runners for fine-tuned Jina and RoBERTa, official Ditto, MixDA, out-of-fold fusion, and joint neural HEF variants.
- Compact public result summaries, per-run metrics, means, standard deviations, and manifests under [`results/`](results/).
- Compatibility runners for official Ditto and AnyMatch baselines.
- The LaTeX manuscript source and bibliography.
- Integrity checks that prevent degenerate scores, incomplete coverage, and unstable tie handling from being published.

Raw datasets, generated embeddings, model checkpoints, and large prediction files are not committed. Their expected locations are documented in [`data/README.md`](data/README.md) and [`artifacts/README.md`](artifacts/README.md).

## Model identities

These systems must not be conflated:

| Name | Role |
|---|---|
| HEF + E5 | Primary frozen-embedding fusion system using `intfloat/e5-base-v2` |
| HEF + frozen RoBERTa | HEF sensitivity system using `sentence-transformers/all-roberta-large-v1` |
| Tuned RoBERTa | Pairwise cross-encoder comparator using `FacebookAI/roberta-base` |
| Official Ditto | Separate Transformer entity-matching comparator |
| AnyMatch | Separate efficient zero-shot comparator |
| Jina reranker | Neural reranker/classifier comparator with validation-fitted threshold |
| Joint neural HEF | End-to-end trainable neural fusion of structured, rule, availability, and semantic evidence |
| HEF-GBDT + neural OOF | Leakage-safe stacking using record-grouped out-of-fold neural scores |

## Quick start

Python 3.11 is the tested baseline.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
paper1-hef --help
pytest
```

For GPU experiments:

```bash
python -m pip install -e '.[gpu,dev]'
```

Place public datasets under `data/extracted/` according to the configuration, then validate the locked splits:

```bash
paper1-hef --project-root . validate --dataset exp01_all
```

Example frozen-embedding HEF run:

```bash
paper1-hef --project-root . encode \
  --dataset abt_buy \
  --model intfloat/e5-base-v2 \
  --device cuda

paper1-hef --project-root . exp01 \
  --dataset abt_buy \
  --model intfloat/e5-base-v2
```

To substitute frozen RoBERTa inside HEF, keep the same fusion pipeline and change only the backbone:

```bash
paper1-hef --project-root . encode \
  --dataset abt_buy \
  --model sentence-transformers/all-roberta-large-v1 \
  --device cuda

paper1-hef --project-root . exp01 \
  --dataset abt_buy \
  --model sentence-transformers/all-roberta-large-v1
```

## Reproducibility rules

- Official dataset splits are immutable.
- Model selection and thresholds use validation data only; test data remains untouched until final evaluation.
- Learned methods use seeds `20260725`, `20260726`, and `20260727`; these are fixed integers, not execution dates selected after seeing results.
- Ranking reports conditional reranking metrics and end-to-end candidate-pool coverage separately, including Hits@100.
- Ties are resolved by score descending, then original retrieval rank ascending, with stable sorting.
- A run is complete only when metrics parse, row/count checks pass, outputs are non-degenerate, and the artifact manifest is verified.

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the full workflow and [`docs/EXPERIMENT_STATUS.md`](docs/EXPERIMENT_STATUS.md) for the release snapshot.


