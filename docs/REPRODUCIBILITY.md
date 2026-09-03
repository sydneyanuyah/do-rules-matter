# Reproducibility workflow

## 1. Freeze the inputs

Record dataset version, checksum, license, official split, model identifier, immutable revision, configuration hash, package environment, and compute type. Never choose a seed from a calendar date after observing results; this project preregisters the fixed integer seeds `20260725`, `20260726`, and `20260727`.

## 2. Validate data

Check required columns, labels, split identity, record/entity leakage, duplicates, and query/candidate alignment. A small number of malformed Link-Lives rows should be isolated and reported with reason codes; they must not silently invalidate or discard the remaining benchmark.

## 3. Train and select

Train only on training data. Select thresholds, prompt variants, backbones, and hyperparameters on validation data. Test data is evaluated only after choices are locked.

## 4. Evaluate

Classification reports F1, precision, recall, average precision, ROC AUC, Brier score, and calibration. Ranking reports pool coverage, conditional MRR/NDCG/Hits, and end-to-end Hits@20/50/100. Ranking ties use score descending and original retrieval rank ascending.

## 5. Aggregate

Report all seed-level values plus mean and standard deviation. Use paired bootstrap units appropriate to the task: test pairs or entity groups for classification and queries for ranking. Report 95% confidence intervals, paired differences, effect sizes, and Holm-adjusted p-values within declared hypothesis families.

## 6. Verify and publish

Assert exact expected row counts, unique pair/query-candidate keys, finite scores, non-degenerate score distributions, both predicted classes where applicable, and successful exits. Synchronize scores, metrics, logs, models, manifests, and commands. Re-download or list remote objects to verify the transfer.

Cloud synchronization is optional and must use environment variables. Never hard-code internal accounts, buckets, credentials, or presigned URLs in public source.
