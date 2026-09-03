# Fine-tuned Jina — Experiment 1

This package implements the missing **Jina fine-tuned** baseline for all six
Experiment 1 datasets and the three locked protocol seeds (`20260725–20260727`).

## Scientific protocol

- Base model: `jinaai/jina-reranker-v2-base-multilingual`
- Pinned revision: `9cfeff2df7d40d1b78e75e5e9cebec92a99813c9`
- Objective: `BCEWithLogitsLoss` on the model's single relevance logit
- Model selection: learning rate, epoch, and decision threshold on validation only
- Test policy: the test split is scored once, after checkpoint and threshold lock
- Replication: one independent run for each of three locked seeds
- GPU policy: longest-pair forward/backward calibration, targeting 20 GiB on A10G
- Isolation: one dataset/seed cell per GPU; atomic output and exclusive lock prevent collisions

## Install in a GPU workspace

The runner uses the repository root by default. A separate workspace can be selected with `PAPER1_ROOT`:

```text
/workspace/paper1-hef/scripts/expanded/exp01_jina
```

The app must also have the canonical `src/`, `configs/experiment.yaml`, and the
dataset selected for its lane under `data/`.

## One GPU / one cell

```bash
PAPER1_ROOT="$PWD" \
bash scripts/expanded/exp01_jina/launch_one.sh \
  abt_buy 20260725 0
```

## Four-GPU host

`launch_matrix.sh` starts the first four missing cells only. Use explicit
`launch_one.sh` commands when coordinating multiple isolated apps so that each
cell is assigned exactly once.

```bash
bash scripts/expanded/exp01_jina/launch_matrix.sh 0,1,2,3
```

## Full 18-cell dispatch plan

With eighteen available GPUs, assign the Cartesian product directly:

```text
abt_buy:              seeds 20260725, 20260726, 20260727
amazon_google:         seeds 20260725, 20260726, 20260727
walmart_amazon:        seeds 20260725, 20260726, 20260727
wdc_80_medium_seen:    seeds 20260725, 20260726, 20260727
wdc_80_medium_unseen:  seeds 20260725, 20260726, 20260727
link_lives_release2:   seeds 20260725, 20260726, 20260727
```

Outputs are written cell-by-cell to:

```text
artifacts/exp01_jina_finetuned/v1/{dataset}/seed_{seed}/
```

A cell is complete only when `metrics.json`, `scores.npz`,
`run_manifest.json`, the saved model, and `COMPLETED.json` exist locally and
the required verification objects can be read from the configured artifact store.
