# Link-Lives supplemental-validation runbook

Link-Lives is not part of Experiment 1. It is a separately labeled public
genealogy transfer study.

## Scope

Release 2 census-to-census records and the domain-expert benchmark only.
Permission-controlled parish-register source files are excluded. Algorithmic
`links` and `life courses` files are never used as gold labels.

## Phase A — immutable acquisition

```bash
bash scripts/download_link_lives_release2.sh
```

Required outputs:

- `14001.zip`
- `14001.zip.sha256`
- `14001.inventory.txt`
- the official Release 2 guide and source manifest

The ZIP must pass `unzip -t` before upload or extraction.

## Phase B — schema and decision audit

```bash
python scripts/audit_link_lives_release2.py \
  data/public_genealogy/link_lives/raw/release_2/14001.zip
```

This extracts only:

- `benchmark v1.xlsx`;
- harmonized census CSVs;
- ALA census candidate files;
- Link-Lives synonym catalogues.

It reports workbook sheets, columns, all decision/agreement/resolution values,
row counts, archive member sizes, CRCs, and selected source files. It does not
create binary labels.

## Phase C — label policy lock

Before building pairs, commit a label-policy manifest that:

1. maps only unambiguous expert decisions to positive or negative;
2. preserves `maybe`, disagreement, and contested decisions as separate
   uncertainty strata or excludes them from the primary binary task;
3. records whether `id2` is absent for no-link decisions;
4. prohibits automatic Link-Lives/XGBoost links from entering gold labels;
5. reports counts before and after every filter.

## Phase D — pair construction

- Join `id1` to its public source record and positive `id2` to the target
  census using `(source, pa_id)`.
- Reconstruct plausible nonmatches from the ALA candidate set or reproduce the
  documented blocking policy.
- Never sample globally random negatives for the headline evaluation.
- Store the candidate-generation version and candidate recall separately.

## Phase E — leakage-safe split

- The outer grouping key is `linking unit`.
- A linking unit cannot occur in more than one split.
- Historical person identifiers and connected components are checked for
  cross-split overlap after pair construction.
- The 1787–1845 benchmark, documented as test-only by Link-Lives, remains an
  external temporal test and is never used for training or tuning.
- Train/validation/test construction for 1845–1901 is locked before features
  or model outcomes are inspected.

## Phase F — Exp01 systems and outcome

Run rules, frozen embedding, manual/equal fusion, validation-tuned convex
fusion, HEF-Linear, HEF-GBDT, and the cross-encoder on identical pairs.
Match-class F1 is primary. Report calibration, paired bootstrap intervals,
missing-evidence slices, and Holm-corrected headline comparisons.
