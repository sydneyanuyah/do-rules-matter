# Baseline strengthening protocol

Status: locked before the new baseline test results are examined.

## Required executed baselines

### Ditto-style RoBERTa with MixDA

The existing `FacebookAI/roberta-base` pair classifier is a high-capacity
cross-encoder reference, but it is not called Ditto. It uses field-aware pair
serialization and fine-tuning, but it does not reproduce all Ditto
optimizations.

A new baseline will use the same RoBERTa checkpoint and splits with
Ditto-style serialization plus MixDA augmentation. The augmentation operator,
rate, random seed, training examples, selected epoch, and validation-selected
threshold are recorded. It is labeled **Ditto-style RoBERTa + MixDA**, not
official Ditto, unless the official Megagon implementation runs without
scientific modification.

Required coverage:

- Experiment 1: all six public datasets and three seeds.
- Experiment 2: score each fixed E5 top-100 pool; report conditional and
  end-to-end ranking metrics.
- Experiment 5A: the same 24 nested label fractions and three seeds.

Reference implementation:
`https://github.com/megagonlabs/ditto`

### AnyMatch

AnyMatch is evaluated in its intended zero-shot setting. For each target
dataset, no labeled pair from that target may enter fine-tuning, example
selection, threshold selection, or calibration. The official leave-one-dataset-
out implementation and GPT-2 base model are used when compatible with the
public inputs.

Required coverage:

- Experiment 1: zero-shot pair-classification F1 on all six targets.
- Experiment 2: zero-shot scoring of the same fixed E5 candidate pools.
- Experiment 5A: one zero-label reference point; it is not placed on the
  supervised percentage curve as if it used target labels.
- Experiment 8: throughput, model size, device, and memory.

Reference implementation:
`https://github.com/Jantory/anymatch`

## Prompted LLM baseline

The Qwen3.5, Ministral 3 8B, Phi, and Qwen3-1.7B runs are called
**MatchGPT-style open-model prompting**. They are not called MatchGPT because
the original hosted models and full protocol are not being reproduced.

The locked grid contains:

- direct and rationale-assisted output modes;
- zero, one, two, three, and four demonstrations;
- deterministic decoding;
- validation-only prompt selection per model;
- strict machine parsing of the final `{"match": 0|1}` field;
- invalid outputs counted as errors;
- final test evaluation performed only after prompt selection.

Rationale-assisted prompting is reported as a request for a brief evidence
sentence, not as a claim that hidden chain-of-thought is observed or required.

The method is anchored to the MatchGPT literature and official prompt/code
repository:
`https://github.com/wbsg-uni-mannheim/MatchGPT`

## Discussed but not reproduced

- **ComEM** is cited for match/compare/select strategies that exploit
  candidate-set interactions. Pairwise HEF reranking does not claim to
  reproduce ComEM.
- **CaRL-EM** is cited for cost-aware sequential routing among LLM operators
  and capacities. The proposed HEF/industrial controller is distinguished by
  field-evidence availability, structured missingness, local rule evidence,
  and non-LLM fusion; it is not claimed as the first generic cost-aware router.
- DeepMatcher, HierMatcher, Unicorn, PromptEM, Sudowoodo, DAME, SETEM, Dedupe,
  Splink, pyJedAI, and Magellan are positioned as prior model families or
  operational systems. They are not silently counted as executed baselines.

## Experiment-specific interpretation

- Experiment 1 compares pair-classification accuracy and calibration.
- Experiment 2 identifies the exact pool and learner in every row. Candidate
  pools are backbone-specific; conditional MRR is accompanied by PoolHit@100
  and end-to-end MRR.
- Experiment 3 separates scalar score-fusion baselines from the clean
  full-feature learner comparison.
- Experiment 4 is called robustness across frozen backbones because the fusion
  layer is retrained for every backbone.
- Experiment 5A contains supervised label-efficiency curves. Experiment 5B
  contains zero/few-shot prompted LLM results. The two are not merged into one
  notion of label efficiency.

