# Experiment 3: full 39-configuration matrix

This is a **locked-output evaluation**, not a second training pass. It consolidates the
validation-selected, untouched-test predictions already produced under Experiments 1
and 2. This prevents duplicate training and test leakage while providing a single
classification/ranking comparison across all six public datasets.

Run:

```bash
python run_exp03_full_matrix.py \
  --artifacts-root /path/to/metadata_snapshot \
  --output-dir artifacts/exp03_full_matrix
```

Use `--strict` for the final publication gate. Standard deviation is the sample SD
over the three fixed seeds. Deterministic one-run methods report a blank SD, not a
fabricated zero.
