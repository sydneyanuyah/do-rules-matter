# Exp2 fine-tuned Jina and HEF-GBDT + tuned-Jina

Each dataset/seed cell runs independently on one GPU. The runner first selects
Jina learning rate and epoch using validation conditional MRR. It then creates
five-fold query-record-grouped OOF scores for every train-pool row, fits the
HEF-GBDT using E5/rule/structured features plus only those OOF Jina scores, and
finally touches test once. Both conditional and end-to-end Hits@100 are stored.

```bash
bash scripts/expanded/exp02_jina_oof/launch_one.sh \
  abt_buy 20260725 0
```

Map the eighteen rows in `dispatch_plan.csv` to eighteen GPUs, or queue them
without duplication across fewer GPUs. S3 conditional locks prevent two apps
from owning the same cell.
