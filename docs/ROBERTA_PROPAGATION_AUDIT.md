# Frozen RoBERTa HEF propagation audit (Experiments 3–9)

Audit date: 2026-08-11  
Canonical artifact store inspected read-only; deployment identifiers are omitted
from the public repository.

## Bottom line

The frozen RoBERTa-embedding HEF is **not yet present in every downstream experiment**.

It is complete for pair classification and candidate ranking on all six public datasets:

- Frozen semantic encoder: `sentence-transformers/all-roberta-large-v1`
- Immutable revision: `cf74d8acd4f198de950bf004b262e6accfed5d2c`
- Experiment 1: all six datasets, three HEF seeds, models, scores, metrics, and manifests
- Experiment 2 / backbone-transfer ranking: all six backbone-specific candidate pools and ranking outputs, now in S3

The directory named `artifacts/exp05_roberta` is **not** this system. It is the separately trained `FacebookAI/roberta-base` joint-pair cross-encoder. Treating those files as RoBERTa-embedding HEF would be a model-identity error.

## Model identity: do not merge these labels

| Label | Model | Training/use | HEF? |
|---|---|---|---|
| Frozen RoBERTa-embedding HEF | `sentence-transformers/all-roberta-large-v1` @ `cf74…` | Encodes left and right records independently; cosine similarity is the `embedding_score`, fused with structured evidence by HEF-Linear/GBDT/Rank | **Yes** |
| Tuned RoBERTa cross-encoder | `FacebookAI/roberta-base` @ `e2da…` | Jointly encodes a record pair and is fine-tuned as a classifier/reranker | **No; comparator** |
| Official/official-style Ditto | RoBERTa pair encoder plus Ditto protocol/augmentations | Joint pair classifier | **No; comparator** |

## Existing frozen-RoBERTa HEF results

All figures below were read from the canonical metrics files. Ranking comparisons are within the RoBERTa-specific, label-blind candidate pool; they are not cross-pool comparisons against E5.

| Dataset | Frozen RoBERTa F1 | RoBERTa HEF-GBDT F1 | F1 delta | Frozen RoBERTa conditional MRR@100 | RoBERTa HEF-Rank conditional MRR@100 | MRR delta | PoolHit@100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Abt–Buy | 0.4168 | 0.7148 | +0.2980 | 0.7850 | 0.9255 | +0.1405 | 1.0000 |
| Amazon–Google | 0.4464 | 0.5671 | +0.1208 | 0.8107 | 0.9234 | +0.1127 | 1.0000 |
| Link-Lives | 0.4324 | 0.8514 | +0.4190 | 0.4646 | 0.4634 | **-0.0011** | 0.6668 |
| Walmart–Amazon | 0.3859 | 0.7812 | +0.3953 | 0.8218 | 0.9407 | +0.1189 | 1.0000 |
| WDC seen | 0.3434 | 0.4571 | +0.1137 | 0.4607 | 0.6110 | +0.1503 | 0.9880 |
| WDC unseen | 0.3948 | 0.4981 | +0.1033 | 0.5757 | 0.7169 | +0.1412 | 0.9980 |

Important result: frozen-RoBERTa HEF improves classification F1 on all six datasets, but HEF-Rank is essentially tied/slightly worse than frozen RoBERTa on Link-Lives. Therefore the current manuscript claim that every backbone-transfer ranking delta is positive must not be extended to RoBERTa.

## Validation-only classification-backbone choice for Experiment 7

Comparing E5 and frozen RoBERTa using mean validation F1 only (never test F1), the best HEF is:

| Dataset | Best validation-selected HEF among E5/RoBERTa | Mean validation F1 |
|---|---|---:|
| Abt–Buy | E5 HEF-GBDT | 0.7072 |
| Amazon–Google | RoBERTa HEF-GBDT | 0.6053 |
| Link-Lives | E5 HEF-GBDT | 0.8972 |
| Walmart–Amazon | E5 HEF-GBDT | 0.7619 |
| WDC seen | E5 HEF-GBDT | 0.5427 |
| WDC unseen | E5 HEF-GBDT | 0.5427 |

Thus, for the **classification** side of Experiment 7, if the candidate set is expanded from the preregistered E5-only HEF to E5-or-RoBERTa, validation would select RoBERTa only for Amazon–Google. A full six-dataset RoBERTa robustness run is still useful, but it must be labeled a separate exploratory backbone-sensitivity analysis rather than silently replacing the locked E5 result.

The ranking backbone has not been selected by this audit. It must be selected from validation queries using an end-to-end measure over all validation queries (so different backbone-specific PoolHit rates are not hidden by conditioning), followed by untouched-test robustness. Test MRR values in the table above must not be used for that choice.

## Exact propagation matrix

| Experiment | Is frozen-RoBERTa HEF currently present? | Scientifically required action | Compute needed | Why |
|---|---|---|---|---|
| Exp3: fusion learner ablation | No; current artifact explicitly fixes E5 | **No primary rerun.** Keep E5 because Exp3 asks which learner is needed under a fixed semantic source. Add a scope note. Optional RoBERTa learner ablation must be a separately named exploratory artifact. | None for primary | Changing the backbone inside a learner ablation changes two factors at once. Current code also mixes a supplied classification backbone with a hard-coded E5 ranking path. |
| Exp4: backbone transfer | **Yes in source artifacts**, but missing from the Exp4 paper table/summary | Add RoBERTa as a fifth, post-lock exploratory backbone using the already-complete Exp1 classification and backbone-specific Exp2 ranking metrics. Rebuild summary/table; do not retrain. | Reporting only | Exp4 is exactly the experiment that asks whether HEF transfers across frozen backbones. |
| Exp5: label efficiency | No. `exp05_roberta` is the tuned cross-encoder, not frozen-RoBERTa HEF | Run all 24 fractions × 3 seeds × 6 datasets for HEF-Linear and HEF-GBDT using the frozen RoBERTa score. Preserve the E5 curve. | CPU; embeddings already exist | A claim about the upgraded HEF's label efficiency requires its own curve. |
| Exp6: evidence ablation | No; current output is E5 and paths are not model-scoped | Preserve E5 primary. Run a separate frozen-RoBERTa HEF ablation across all six datasets if the upgraded system is claimed beyond Exp4. Use the RoBERTa-specific ranking pool. | CPU | Evidence importance can depend on semantic-score quality. This is required for an “upgraded HEF” evidence claim, but not to close the original E5 Exp6. |
| Exp7: controlled evidence loss | No; scripts hard-code E5 | Preserve locked E5 headline. For classification, expanded validation selection chooses RoBERTa only on Amazon–Google. Ranking needs a new validation-only, end-to-end backbone selection before any replacement. For a stronger non-cherry-picked audit, run a separate RoBERTa robustness appendix on all six datasets with identical deterministic masks; reuse rules/Ditto results and compute only frozen RoBERTa + RoBERTa HEF. | GPU encoding under masks; CPU HEF scoring | Do not use test results to choose the backbone. Do not aggregate E5 and RoBERTa candidate pools as though they were identical. |
| Exp8: efficiency/deployment | No; GPU profiler hard-codes E5 and CPU output would overwrite E5 | Add model/revision arguments and model-scoped output. Benchmark frozen RoBERTa encoder batch/online throughput, latency, peak GPU memory, and model size on all six test sets. Reuse/recompute structured-feature time and HEF CPU inference in a model-scoped end-to-end ledger. | GPU for timing; CPU for fusion | The upgraded encoder is much larger than E5; accuracy without its measured deployment cost is incomplete. Existing cached embeddings cannot substitute for timing. |
| Exp9: statistics | Classification glob can ingest RoBERTa Exp1 metrics if rerun, but current published artifact predates them; ranking statistics are E5-only | Rerun classification aggregation to include a distinct `backbone=all-roberta-large-v1` Holm family. Add a **secondary within-RoBERTa-pool** paired query bootstrap for HEF-Rank vs RoBERTa embedding/rules/equal/convex. Do not compare RoBERTa-pool ranks directly with E5-pool external rerankers unless those rerankers are scored on the same RoBERTa pool. | CPU | Candidate pools differ by backbone; statistical units must remain aligned within one fixed pool. |

## Mandatory code fixes before downstream runs

The current runners cannot safely be invoked as-is for frozen RoBERTa downstream propagation.

1. Add the frozen model to `configs/experiment.yaml`:

```yaml
- id: sentence-transformers/all-roberta-large-v1
  revision: cf74d8acd4f198de950bf004b262e6accfed5d2c
  pooling: mean
  normalize: true
  symmetric_prefix: ""
  max_length: 256
```

2. Make every downstream output model-scoped. Recommended roots:

```text
artifacts/exp05_by_backbone/sentence-transformers__all-roberta-large-v1/<dataset>/...
artifacts/exp06_by_backbone/sentence-transformers__all-roberta-large-v1/<dataset>/...
artifacts/exp07_by_backbone/sentence-transformers__all-roberta-large-v1/{classification,ranking}/<dataset>/...
artifacts/exp08_by_backbone/sentence-transformers__all-roberta-large-v1/{cpu_fusion,gpu}/<dataset>/...
artifacts/exp09/backbone_secondary/sentence-transformers__all-roberta-large-v1/...
```

3. Fix model/pool coupling:

- Exp3 ranking currently reads `artifacts/exp02/ranking/<dataset>` even when `--model` changes. Do not run RoBERTa Exp3 until it uses `ranking_by_backbone/<slug>/<dataset>` and a model-scoped output.
- Exp6 currently reads embeddings for `--model` but reads the E5 pool from `artifacts/exp02/candidate_pools/<dataset>`. For RoBERTa it must read `candidate_pools_by_backbone/<slug>/<dataset>`.
- Exp7 classification and ranking hard-code `MODEL_ID`, `MODEL_REVISION`, E5 model selection, E5 pools, and non-scoped outputs. Parameterize all of them.
- Exp8 GPU hard-codes E5; Exp8 CPU accepts `--model` but writes a non-scoped output. Parameterize and scope both.
- Add output assertions that `metrics.json` records the requested model ID/revision and that candidate-pool manifests match them.

4. Preserve the existing E5 artifacts. No runner may overwrite `artifacts/exp03`, `exp05`, `exp06`, `exp07`, or `exp08` while producing RoBERTa results.

## Exact run plan after patching

The commands below assume patched model-scoped runners and the canonical NVMe root.

```bash
export PAPER1_ROOT=/absolute/path/to/Paper1-HEF-GitHub
export PAPER1_MODEL=sentence-transformers/all-roberta-large-v1
export PAPER1_REV=cf74d8acd4f198de950bf004b262e6accfed5d2c
cd "$PAPER1_ROOT"
export PYTHONPATH="$PAPER1_ROOT/src"
```

### Exp4 (report-only)

```bash
python scripts/summarize_exp04_backbones.py \
  --include-model "$PAPER1_MODEL" --revision "$PAPER1_REV" \
  --classification-root artifacts/exp01_final \
  --ranking-root artifacts/exp02/ranking_by_backbone
```

This summarizer does not currently exist; it is the smallest required Exp4 code addition.

### Exp5 (six independent CPU lanes)

```bash
for dataset in abt_buy amazon_google link_lives_release2 walmart_amazon wdc_80_medium_seen wdc_80_medium_unseen; do
  nohup python -m paper1_hef.cli --project-root . exp05 \
    --dataset "$dataset" --model "$PAPER1_MODEL" --revision "$PAPER1_REV" \
    --bootstrap-replicates 2000 \
    > "artifacts/logs/exp05_frozen_roberta_hef_${dataset}.log" 2>&1 &
done
```

### Exp6 (six independent CPU lanes)

```bash
for dataset in abt_buy amazon_google link_lives_release2 walmart_amazon wdc_80_medium_seen wdc_80_medium_unseen; do
  nohup python scripts/run_exp06.py --project-root . --dataset "$dataset" \
    --model "$PAPER1_MODEL" --revision "$PAPER1_REV" \
    > "artifacts/logs/exp06_frozen_roberta_hef_${dataset}.log" 2>&1 &
done
```

### Exp7 (separate RoBERTa sensitivity analysis)

```bash
for dataset in abt_buy amazon_google link_lives_release2 walmart_amazon wdc_80_medium_seen wdc_80_medium_unseen; do
  CUDA_VISIBLE_DEVICES=$((i % 4)) nohup python scripts/run_exp07_classification.py \
    --project-root . --dataset "$dataset" --model "$PAPER1_MODEL" --revision "$PAPER1_REV" \
    > "artifacts/logs/exp07_class_frozen_roberta_hef_${dataset}.log" 2>&1 &
  i=$((i + 1))
done
```

Run ranking similarly only after the runner selects the RoBERTa-specific fixed pool and RoBERTa-specific clean-validation model. A more efficient implementation should reuse the existing deterministic masks and existing rules/official-Ditto scores, and compute only `frozen_embedding` and `best_hef` for the RoBERTa appendix.

### Exp8 (GPU lanes, measured rather than inferred)

```bash
for dataset in abt_buy amazon_google link_lives_release2 walmart_amazon wdc_80_medium_seen wdc_80_medium_unseen; do
  CUDA_VISIBLE_DEVICES=$((i % 4)) nohup python scripts/run_exp08_gpu.py \
    --project-root . --dataset "$dataset" --model "$PAPER1_MODEL" --revision "$PAPER1_REV" \
    --embedding-batch-size 256 --repeats 5 --online-pairs 100 \
    > "artifacts/logs/exp08_gpu_frozen_roberta_${dataset}.log" 2>&1 &
  i=$((i + 1))
done

for dataset in abt_buy amazon_google link_lives_release2 walmart_amazon wdc_80_medium_seen wdc_80_medium_unseen; do
  python scripts/run_exp08_cpu.py --project-root . --dataset "$dataset" \
    --model "$PAPER1_MODEL" --revision "$PAPER1_REV"
done
```

Batch 256 is only an initial probe. The production runner should adapt upward/downward after a brief `nvidia-smi` measurement while preserving the measurement protocol.

### Exp9

```bash
python scripts/run_exp09_backbone_secondary.py \
  --classification-root artifacts/exp01_final \
  --ranking-root "artifacts/exp02/ranking_by_backbone/sentence-transformers__all-roberta-large-v1" \
  --model "$PAPER1_MODEL" --revision "$PAPER1_REV" \
  --replicates 10000 --seed 20260725 \
  --output-dir "artifacts/exp09/backbone_secondary/sentence-transformers__all-roberta-large-v1"
```

`run_exp09_backbone_secondary.py` does not currently exist. It must calculate conditional MRR@100 per query from the RoBERTa pool and retain only within-pool aligned hypotheses.

## Minimal completion criteria

- Exp4 table includes RoBERTa classification and ranking, model revision, six datasets, and explicitly marks it as a post-lock exploratory addition.
- Exp5 contains exactly `6 × 24 × 3 × 2 = 864` frozen-RoBERTa HEF learner fits, all nonempty, aligned, and model-scoped.
- Exp6 contains all 10 conditions (full, seven group removals, raw-only, aggregate-rule-only), both classification learners, both ranking learners, three seeds, and six datasets under the RoBERTa model scope.
- Exp7 preserves E5 headline results; any RoBERTa appendix uses identical deterministic masks, clean-validation-only selection, and a RoBERTa-specific fixed ranking pool.
- Exp8 reports encoder + structured feature construction + fusion inference end-to-end, with batch and online numbers and actual peak memory.
- Exp9 uses separate Holm families by backbone/pool and never compares misaligned candidate sets.
- Every result manifest records model ID, immutable revision, dataset/split hashes, seeds, candidate-pool manifest, and command/code hash.

## Recommended scientific decision

Do not rename frozen RoBERTa as the new primary HEF on the basis of test results. Preserve E5 as the preregistered headline system. Add frozen RoBERTa as a fifth, explicitly post-lock backbone-transfer and robustness/efficiency sensitivity analysis. This uses all completed RoBERTa work, answers whether it improves HEF, and avoids outcome-driven replacement of the primary model.
