# Exp1 HEF-GBDT + E5 + task-trained neural evidence

This package closes two Experiment 1 combinations:

- HEF-GBDT + E5 + tuned RoBERTa
- HEF-GBDT + E5 + fine-tuned Jina

For every dataset, neural family, and protocol seed, the training pairs are
partitioned into three connected components over both record sides. Each fold
model scores only its held-out components. These OOF scores are appended to the
locked structured+E5 HEF features. Validation and test neural scores are read
from the already completed full-training Exp1 artifact for the same
dataset/seed; exact pair-ID and label alignment is mandatory.

HEF capacity and classification threshold are selected only on validation F1.
Test is scored once after selection. All output paths are unique by dataset,
neural family, and seed and are committed atomically.

One cell:

```bash
bash run_cell.sh abt_buy roberta 20260725 0
bash run_cell.sh abt_buy jina 20260725 0
```

Run the CPU-only admission check before allocating GPUs:

```bash
PYTHONPATH="$PWD/src" \
python preflight_all.py --project-root "$PWD"
```

Schedules:

- `schedule_18gpu.tsv`: 18 lanes, one dataset/seed per GPU, RoBERTa then Jina.
- `schedule_36gpu.tsv`: maximum parallelism, one independent cell per GPU.
