# Protocol lock — pre-GPU checklist

Status on 2026-07-26: **Experiments 1 and 2 are authorized**.

## Locked

- All empirical claims are supported by public benchmarks.
- Private genealogy is use-case context only and is excluded from model
  selection, pooled metrics, significance tests, and generality language.
- Compute location and artifact storage are deployment details supplied through
  environment variables; they are not part of the scientific protocol.
- Public standard datasets: Abt–Buy, Amazon–Google, Walmart–Amazon, and WDC
  Products.
- Exp01 headline datasets: WDC 100%-unseen, Abt–Buy, and Amazon–Google.
- Exp01 secondary controls: WDC 0%-unseen and Walmart–Amazon.
- Link-Lives is a separately reported public-genealogy comparison family in
  Experiment 1 and is never pooled silently with product-matching datasets.
- WDC pairwise formulation and 80%-corner-case archive.
- Official train/validation/test files only.
- Symmetric record serialization with explicit field markers.
- Train-only preprocessing and normalization.
- Validation-only model, hyperparameter, calibration, and threshold selection.
- Test evaluation after locking.
- Seeds: 20260725, 20260726, 20260727.
- Candidate recall is reported separately from conditional reranking quality.
- DBLP–ACM is not used to support generality claims.

## Resolved for Experiment 1

- The primary backbone is E5-base-v2 at immutable revision
  `f52bf8ec8c7124536f0efb74aca902b2995e5bcd`.
- The cross-encoder is RoBERTa-base at immutable revision
  `e2da8e2f811d1448a5b465c236feacd80ffbac7b`.
- The cross-encoder search budget is learning rates
  `{1e-5, 2e-5, 3e-5}`, at most 20 epochs, and validation-only early stopping
  with patience 5.
- Experiment 1 uses binary-log-loss HEF-Linear and HEF-GBDT. Ranking objectives
  and Exp12C are outside Experiment 1.
- An MLP is not a required Experiment 1 system.
- The model revisions were rechecked against the official Hugging Face model
  API immediately before the run.
- The controlled WDC primary/secondary pair is confirmed.
- The WDC 100%-unseen condition enforces zero offer overlap. The 0%-unseen
  control intentionally permits previously seen offers; its observed overlap
  is reported rather than misclassified as leakage.

## Still blocked for later experiments

- Freeze the selected best HEF using unmasked validation performance before
  Experiment 7.

## Resolved for Experiment 2

- Candidate generation is label-blind dense retrieval with the locked E5
  revision used by Experiment 1.
- Candidate pool sizes are K = 20, 50, and 100.
- PoolHit@K is reported separately from conditional reranking quality.
- Conditional metrics are MRR, Hits@1/5/10, and nDCG.
- HEF-Linear and HEF-GBDT are pointwise ranking baselines over identical
  evidence. HEF-Rank uses a true query-grouped LambdaMART objective.
- DeepMatcher catalog queries receive deterministic ID-hash group splits so a
  query cannot cross train, validation, and test. WDC retains its official
  train/validation/test files; its unseen test condition is the headline.
- The candidate generator never sees match labels. Labels are joined only
  after the top-100 retrieval list is frozen.
- The public scope contains the five product benchmarks and separately reported
  Link-Lives genealogy benchmark. Private genealogy remains excluded.

## Paper eligibility rule

A run becomes paper-eligible only when:

1. its manifest contains immutable dataset hashes, model revision hashes, code
   hash, environment hash, hardware, seed, and correction-family ID;
2. all three seeds finish;
3. no preprocessing or selection used the test split;
4. the exact command succeeds from a clean environment; and
5. the result table is generated from manifests rather than copied manually.
