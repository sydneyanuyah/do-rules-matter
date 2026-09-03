# Revised Experiments 7, 8, and 9

## Locked comparison set

The comparison contains exactly ten rerankers: four standalone neural systems,
two HEF-Linear systems, two HEF-GBDT systems, and two jointly trained neural HEF
systems. The registry records their exact upstream artifact families. Selection
is frozen before any masked test result is inspected.

## Experiment 7: controlled evidence loss

The six public test candidate pools from Experiment 2 remain fixed. Evidence is
masked at record level, so every occurrence of one record receives the same
mask. The same mask realization is reused for all ten systems. The clean level
is evaluated once; 10%, 30%, 50%, and 70% are each evaluated with the three
fixed seeds. This gives 13 scenarios per model and dataset, or 780 total
model-scenarios.

Models are never retrained or retuned on masked test evidence. Primary outputs
are conditional MRR@100 and its retention relative to the clean condition.
Hits@1 and Hits@1 retention are secondary. End-to-end pool coverage is retained
as a required diagnostic.

## Experiment 8: efficiency and deployment

Experiment 8 instruments the same calls used by Experiment 7. It reports:

- cold end-to-end time, including checkpoint/tokenizer load, serialization,
  semantic inference, feature construction, fusion, and ranking;
- model-loaded uncached inference time;
- warm/cached reranking time, where cached semantic scores/features are valid;
- latency distributions, throughput, peak CPU RSS, peak allocated/reserved GPU
  memory, serialized model size, and parameter footprint;
- upstream training time/cost from canonical manifests separately from the
  cost of the robustness evaluation.

Candidate retrieval, S3 transfer, and dataset download are reported separately
and are not hidden inside reranking latency. A metric that does not apply to a
standalone model is recorded as null, never zero.

## Experiment 9: statistical validation

Experiment 9 is blocked until all Experiment 7 per-query outputs pass row,
coverage, score, and synchronization checks. Paired bootstrap sampling is over
queries, retaining the matched mask realization and candidate pool. It reports
95% confidence intervals, paired differences, standardized paired effect size,
two-sided raw p-values, and Holm-adjusted p-values for the registered primary
comparisons.

The six public datasets form the hypothesis families in this release. Missing
per-query score artifacts block only the affected public comparison and do not
invalidate unrelated dataset families.
