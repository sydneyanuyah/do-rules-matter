# Data layout

Only public datasets are in scope. Download each dataset from its official source and comply with its license; raw data is deliberately not distributed in this repository.

The default configuration expects:

```text
data/
└── extracted/
    ├── deepmatcher/
    │   ├── abt_buy/exp_data/
    │   ├── amazon_google/
    │   └── walmart_amazon/exp_data/
    ├── wdc/80pair/
    └── ../public_genealogy/link_lives/processed/exp01/
```

Each prepared pair split must retain its official train/validation/test identity. Run `paper1-hef --project-root . validate --dataset exp01_all` before any experiment. See `docs/DATA_SOURCES.md` and the Link-Lives preparation scripts for provenance and schema details.
