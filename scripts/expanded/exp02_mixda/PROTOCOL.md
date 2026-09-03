# HEF-GBDT + MixDA-style augmentation protocol

This runner adapts the already implemented Ditto-style `mixda_augment` text
operators to HEF. It does **not** claim to reproduce official Ditto.

For each locked protocol seed, the runner:

1. loads and validates the official train/validation/test split;
2. creates one deterministic augmented copy of every training pair using the
   `del`, `swap`, `drop_col`, and `append_col` operator family;
3. parses the augmented COL/VAL text back into the dataset's field schema;
4. reports malformed/unknown fields and falls back only when an augmented side
   becomes empty, preserving every training row and label;
5. recomputes both structured features and the locked-backbone cosine score for
   augmented training rows;
6. fits a fixed HEF-GBDT on original plus augmented training features;
7. selects only the classification threshold on validation; and
8. evaluates the untouched test set once after threshold lock.

Validation and test are never augmented. Existing original dense artifacts are
checked for exact pair-ID and label alignment. Each dataset/backbone/seed has an
independent collision-safe output directory. A combination is complete only
after three seed metrics, models, aligned score arrays, aggregate metrics, and
scoped S3 synchronization exist.

The paper-facing label must be **HEF-GBDT + MixDA-style field augmentation with
{backbone}**. It must not be shortened to “official MixDA” or “Official Ditto”.
