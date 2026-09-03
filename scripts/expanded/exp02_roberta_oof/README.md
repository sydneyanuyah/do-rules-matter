# Exp2 HEF-GBDT + tuned RoBERTa evidence

This runner propagates the Experiment 1 tuned-RoBERTa combination into the
frozen E5 top-100 candidate pools used by Experiment 2.

For each dataset and protocol seed it:

1. reads the already selected RoBERTa learning rate and epoch from the matching
   Experiment 1 result;
2. partitions the Exp2 training queries into three deterministic folds;
3. trains one RoBERTa model per complement and scores only held-out query
   records, producing complete OOF training scores without in-sample stacking;
4. trains one full Exp2 model and scores validation candidates;
5. fits HEF-GBDT candidates using structured evidence, E5 score, and OOF
   RoBERTa score;
6. selects the GBDT capacity using validation conditional MRR@100;
7. scores the untouched test candidates once and reports end-to-end and
   conditional MRR, NDCG, Hits@1/5/10/100, and retrieval coverage.

Candidate records intentionally form a shared retrieval catalogue, so grouping
both query and candidate IDs would collapse the pool graph into a giant
component. The leakage unit is therefore the query record, matching the locked
Exp2 query-grouped split and ranking evaluation unit. Each query occurs in
exactly one OOF fold and never occurs in the corresponding fold-training data.

Run one cell:

```bash
bash run_cell.sh abt_buy 20260725 0
```

Run one durable dataset lane per GPU:

```bash
PAPER1_HOST=Paper-1-Exp2-A bash launch_host_lanes.sh schedule_6gpu.tsv
```

`schedule_6gpu.tsv` requires six GPUs and runs the three seeds sequentially per
dataset. `schedule_18gpu.tsv` is the maximum-parallel schedule: one independent
dataset/seed cell per GPU. Use it when quota permits and the 12-hour wall-clock
target is more important than instance cost.
