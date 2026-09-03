# Data sources and provenance

## Public benchmarks downloaded

| Dataset | Original source | Local archive | SHA-256 |
|---|---|---|---|
| Abt–Buy | DeepMatcher experimental data, UW–Madison | `data/raw/deepmatcher/abt_buy_exp_data.zip` | `cd806d925aca21e6ab942b239353a10ee88904c59d3d3e3da6a74e364e9200a1` |
| Amazon–Google | DeepMatcher experimental data, UW–Madison | `data/raw/deepmatcher/amazon_google_exp_data.zip` | `9b19c85c98c6289f38969e4961cb944bdf986123fc42f6bf17283ac6eb502e61` |
| Walmart–Amazon | DeepMatcher experimental data, UW–Madison | `data/raw/deepmatcher/walmart_amazon_exp_data.zip` | `d6604b0562c2ef3afec8186d7b43e8bdd82d68325bb98b75bbd01f468d4aa57e` |
| WDC Products 80% pairwise | Web Data Commons / University of Mannheim | `data/raw/wdc/80pair.zip` | `b2044939cee5ea6f12148a2f3551508de3cb77660dfc91767c44daaf9d8a9c4a` |

DeepMatcher's dataset index documents the 3:1:1 split construction and reports:
Abt–Buy 9,575 pairs / 1,028 positives; Amazon–Google 11,460 / 1,167;
Walmart–Amazon 10,242 / 962.

WDC Products documents 11,715 offers representing 2,162 entities, nine train,
nine validation, and nine test sets combinable into 27 variants. Offers are
non-overlapping across splits. The 80% archive contains pairwise train and
validation files for small/medium/large development sizes and test files for
0%, 50%, and 100% unseen entities.

## Source URLs

- DeepMatcher dataset index:
  https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md
- WDC Products benchmark:
  https://webdatacommons.org/largescaleproductcorpus/wdc-products/
- WDC Products paper:
  https://arxiv.org/abs/2301.09521
- Ditto reference implementation:
  https://github.com/megagonlabs/ditto
- EACL 2027 Industry Track call:
  https://2027.eacl.org/calls/industry/

## Public genealogy shortlist

The source-verified, admission-gated shortlist is Census Tree, LIFE-M V5, and
Link-Lives Release 2. See `docs/PUBLIC_GENEALOGY_DATASETS.md` for scale,
access conditions, exact experimental role, and unresolved gates. These
datasets are described as public-use/open-access unless their release terms
explicitly establish public-domain status.
