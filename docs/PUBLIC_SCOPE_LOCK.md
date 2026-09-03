# Public scope lock

Effective 2026-07-25, the paper's empirical contribution is based entirely on
public benchmarks.

## Claim-bearing datasets

### Headline

1. WDC Products: 80% corner cases, medium development set, 100% unseen test
   entities.
2. Abt–Buy on the official DeepMatcher split.
3. Amazon–Google on the official DeepMatcher split.

### Secondary

1. WDC Products matched control: 80% corner cases, medium development set,
   0% unseen test entities.
2. Walmart–Amazon on the official DeepMatcher split.

DBLP–ACM is excluded from generality claims because it is a saturated
diagnostic benchmark.

## Private genealogy boundary

Private genealogy is not an experiment in the claim-bearing paper package. It
may be described only as a motivating or deployment use case. It cannot:

- enter a headline or appendix result table;
- enter a cross-domain average;
- determine the best model, learner, threshold, feature set, or backbone;
- enter a confidence interval, hypothesis test, or Holm family;
- support a claim of generalization, robustness, or label efficiency; or
- be required to reproduce any public result.

If included, the vignette should describe why structured evidence and missing
fields matter operationally, without disclosing private records or presenting
unverifiable performance as scientific evidence.

