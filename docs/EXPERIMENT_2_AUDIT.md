# Experiment 2 pre-repair canonical-results audit (2026-08-11)

> Historical provenance: the defect described below was subsequently repaired.
> The current release status and repaired counts are recorded in
> `EXPERIMENT_STATUS.md`. Paths to the private artifact store are omitted.

## Verdict

The canonical tree is structurally complete for six datasets and eleven methods,
and all 66 method/dataset metric records include conditional and end-to-end
Hits@100. AnyMatch and tuned-RoBERTa Link-Lives files each have complete,
duplicate-free, finite, exactly aligned 325,000-row coverage (3,250 queries x
100 candidates).

The Link-Lives tuned-RoBERTa result is **not safe to publish or treat as final**,
despite the existing integrity report saying `publishable: true`. Its three seed
vectors and their mean each contain only two distinct values. Those values split
exactly at the inference batch boundary: 323,584 rows (= 79 x 4,096) versus the
final 1,416-row batch. The score is therefore driven by batch size rather than
record content.

The cause is visible in the repair source: `repair_20260808/src/paper1_hef/exp02.py`
lines 864-865 select only product `FIELDS` for Link-Lives genealogy records.
Consequently no genealogy columns are joined and both RoBERTa texts serialize
empty. The bundle source at `exp01_to_exp05_bundle/src/paper1_hef/exp02.py`
lines 913-918 correctly uses the domain-aware `_fields_for_frame` selection.

AnyMatch does not show this defect: it has 325,000 finite scores spanning
8.7955664e-07 to 0.99999934, and its scorer explicitly selects
`GENEALOGY_FIELDS` for Link-Lives.

## Coverage and Hits@100

| Dataset | Methods | E2E Hits@100 | Pool-hit queries / total |
|---|---:|---:|---:|
| abt_buy | 11 | 1.000000 | 154 / 154 |
| amazon_google | 11 | 1.000000 | 163 / 163 |
| walmart_amazon | 11 | 1.000000 | 131 / 131 |
| wdc_80_medium_seen | 11 | 0.998000 | 499 / 500 |
| wdc_80_medium_unseen | 11 | 0.996000 | 498 / 500 |
| link_lives_release2 | 11 | 0.985231 | 3,202 / 3,250 |

All methods have conditional Hits@100 = 1.0 by definition after conditioning on
pool hits. Within each dataset all methods share E2E Hits@100 because they rerank
the same fixed E5 top-100 pool.

Methods present in every dataset: `embedding`, `rules`, `equal_fusion`,
`convex_fusion`, `hef_linear`, `hef_gbdt`, `hef_rank`, `jina_cross_encoder`,
`tuned_cross_encoder`, `ditto_style_roberta_mixda`, and `anymatch_official`.

## Exact sources and generated tables

Canonical table inputs are one `artifacts/exp02/ranking/{dataset}/metrics.json`
file per dataset.

with `{dataset}` equal to `abt_buy`, `amazon_google`, `walmart_amazon`,
`wdc_80_medium_seen`, `wdc_80_medium_unseen`, or `link_lives_release2`.

Canonical Link-Lives evidence consists of aligned AnyMatch, tuned-RoBERTa,
and HEF score tables plus the integrity JSON and row-level integrity report.

The internal audit also generated a 66-row long table and a six-row MRR@100
matrix. They are not distributed because this code-only bundle excludes result
artifacts; public tables should be regenerated from canonical manifests.

The generated tables faithfully reflect current canonical JSON, but are
provisional until Link-Lives RoBERTa is rescored with genealogy fields and the
Link-Lives `tuned_cross_encoder` metrics entry is replaced.

## Required repair

Use the domain-aware field selection already present in the bundle source,
rescore Link-Lives RoBERTa, and strengthen `assert_score_integrity.py` to reject
near-degenerate/low-cardinality vectors and batch-boundary-correlated scores.
Then regenerate the Link-Lives metrics and these consolidation tables.
