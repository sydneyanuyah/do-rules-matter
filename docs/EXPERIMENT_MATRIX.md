# Paper 1 experiment matrix

Nothing in this matrix authorizes an outcome-dependent scope change. Dataset
variants, primary comparisons, seeds, and model revisions are locked before
paper-eligible runs.

| ID | Question | Inputs | Systems | Metrics | Compute | Gate |
|---|---|---|---|---|---|---|
| Exp01 | Does learned fusion improve pair classification? | WDC primary; Abt–Buy; Amazon–Google; Walmart–Amazon; Link-Lives Release 2 genealogy | rules, frozen embedding, linear HEF, GBDT HEF, cross-encoder | match-class F1 primary; precision, recall, AP, AUROC, Brier, ECE | CPU except embedding/CE scoring | MUST |
| Exp02 | Does fusion improve candidate ranking? | Public candidate pools built label-blind | embedding, rules, linear HEF, GBDT HEF, CE | PoolHit@K separately; conditional MRR, Hits@1/5/10, nDCG@K | GPU retrieval/scoring; CPU fusion | SHOULD |
| Exp03 | Which fusion learner is necessary? | Same fixed scores/features as Exp01/02 | manual/equal, validation convex, linear, GBDT | ΔF1, ΔMRR with paired CI | CPU | MUST |
| Exp04 | Does the effect transfer across frozen backbones? | Identical serialized records | E5-base-v2, BGE-base-en-v1.5, GTE-base-en-v1.5 | per-backbone ΔF1/ΔMRR | GPU encode once; CPU reuse | SHOULD |
| Exp05 | How label-efficient is fusion? | Nested, group-aware train subsets | linear, GBDT, cross-encoder where feasible | F1 vs label fraction | CPU; CE GPU | SHOULD |
| Exp06 | Which evidence groups matter? | Fixed full model and feature groups | leave-one-group-out; raw field vs rule score | paired ΔF1/ΔMRR | CPU | MUST |
| Exp07 | How robust is the locked best HEF model to missing evidence? | Random and group-specific masks p={0,.1,.3,.5,.7} | rules, embedding, CE, HEF selected before masking | retention ratio plus absolute metric | CPU using cached scores | MUST |
| Exp08 | What is the deployment tradeoff? | Same fixed hardware and batches | all headline systems | throughput, p50/p95 latency, peak memory, dollar estimate | CPU/GPU | SHOULD |

## Closed primary comparison family

Holm correction family `headline-v1` contains only:

1. HEF-GBDT versus frozen-embedding-only on each MUST dataset.
2. HEF-Linear versus frozen-embedding-only on each MUST dataset.
3. HEF-GBDT versus the tuned cross-encoder on the WDC primary condition.
4. WDC primary unseen versus its matched seen control for each locked system.

All other comparisons are secondary/exploratory and are labeled as such.

## Public-only evidence boundary

- Headline datasets: WDC 80% corner cases / medium development / 100% unseen,
  Abt–Buy, and Amazon–Google.
- Secondary public checks: the matched WDC 0%-unseen control and
  Walmart–Amazon.
- Public genealogy: Link-Lives Release 2 is included in Experiment 1 as a
  distinct genealogy comparison family. Its person-record-disjoint split,
  frozen embeddings, three-seed HEF runs, and three-seed cross-encoder must
  all pass before Experiment 1 is complete.
- DBLP–ACM may be used only as a saturated diagnostic and cannot support a
  generality claim.
- Private genealogy is not part of the experimental matrix. If retained in the
  paper, it is a short use-case vignette describing practical motivation and
  deployment constraints without entering any aggregate, significance test,
  method-selection decision, or reproducibility claim.
- Link-Lives was added to Experiment 1 by a documented scope amendment on
  2026-07-28, before its final embedding or model run. Because the product
  results already existed, its paired tests use a separate `genealogy-v1`
  family and cannot retrospectively change the product comparison family.

## Scope decisions still requiring a human or policy owner

- The final author list.
- The cross-encoder tuning budget and compute ceiling.
- Whether ranking can be completed without weakening the pair-classification
  core.
