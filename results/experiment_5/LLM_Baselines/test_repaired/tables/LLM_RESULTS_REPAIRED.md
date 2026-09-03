# Repaired open-LLM baseline results

All 24 model–dataset cells pass expected-coverage validation. Values are strict match F1; malformed outputs count as errors.

| Model | abt_buy | amazon_google | link_lives_release2 | walmart_amazon | wdc_80_medium_seen | wdc_80_medium_unseen |
|---|---:|---:|---:|---:|---:|---:|
| Ministral 3 8B | 0.728 | 0.528 | 0.714 | 0.545 | 0.767 | 0.795 |
| Phi-4 Mini | 0.655 | 0.536 | 0.443 | 0.506 | 0.597 | 0.657 |
| Qwen3.5 9B | 0.870 | 0.650 | 0.231 | 0.832 | 0.864 | 0.892 |
| Qwen3 1.7B | 0.590 | 0.173 | 0.322 | 0.222 | 0.534 | 0.587 |
