# Official Ditto OOF + HEF fusion

This package generates leakage-safe train-set scores for Official Ditto and
Official Ditto+MixDA, then trains an HEF-GBDT fusion using structured evidence,
the frozen E5 score, and the Ditto score.

The train pairs are partitioned by connected record components. A component is
never split across folds, so neither the left nor right entity of an OOF pair
can occur in the model's fold-training data. Each fold model is trained for the
epoch selected by the already completed full-training Official Ditto run. It
does not inspect its held-out fold labels for checkpoint or threshold selection.
The original validation and test scores come from that completed full-training
run and are checked for exact pair-ID and label alignment.

Outputs are written atomically under:

```
artifacts/exp01_hef_cross_evidence/v1/<dataset>/intfloat__e5-base-v2/<revision>/
  official_ditto_<plain|mixda_all>/seed_<seed>/
```

Run one collision-safe cell:

```bash
bash run_cell.sh abt_buy plain 20260725 0
```

The launcher uses one process per GPU. See `schedule_11gpu.tsv` for the complete
36-cell schedule. `launch_host_lanes.sh` gives every physical GPU a sequential
queue and advances it immediately when its preceding cell finishes; there is no
global wave barrier. `launch_wave.sh` is retained for manual recovery only.

Scientific safeguards:

- exact three-fold connected-component grouping;
- zero record overlap between fold train and fold holdout;
- 100% OOF row coverage, exactly once per train row;
- exact pair-ID and label alignment for E5 and full Ditto scores;
- finite and nonconstant neural scores;
- validation-only HEF threshold selection;
- untouched test scoring after HEF fitting and threshold selection;
- unique seed/variant/dataset paths, exclusive locks, atomic completion marker;
- scoped S3 upload only after local validation succeeds.
