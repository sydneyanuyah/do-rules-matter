# Paper 1: LLM prompt evaluation audit

Source: audited S3 `llm_eval` artifacts. Validation metrics pool the confusion counts across three demonstration seeds for 1–4 shot prompts; 0-shot has one run. Test prompt results were not present in S3 at audit time and are therefore marked **not run or not uploaded**, never as zero.

## Qwen3.5 9B — validation

| Dataset | Prompt | N | Accuracy | Precision | Recall | F1 | Parse | Invalid |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Abt–Buy | Direct 0-shot | 1000 | 0.9810 | 0.9785 | 0.8426 | 0.9055 | 1.0000 | 0 |
| Abt–Buy | Direct 1-shot | 3000 | 0.9797 | 0.8997 | 0.9136 | 0.9066 | 1.0000 | 0 |
| Abt–Buy | Direct 2-shot | 3000 | 0.9797 | 0.9021 | 0.9105 | 0.9063 | 1.0000 | 0 |
| Abt–Buy | Direct 3-shot | 3000 | 0.9823 | 0.8997 | 0.9414 | 0.9201 | 1.0000 | 0 |
| Abt–Buy | Direct 4-shot | 3000 | 0.9790 | 0.8718 | 0.9444 | 0.9067 | 1.0000 | 0 |
| Abt–Buy | Rationale 0-shot | 1000 | 0.9760 | 0.9286 | 0.8426 | 0.8835 | 1.0000 | 0 |
| Abt–Buy | Rationale 1-shot | 3000 | 0.9713 | 0.8814 | 0.8514 | 0.8661 | 0.9997 | 1 |
| Abt–Buy | Rationale 2-shot | 3000 | 0.9570 | 0.9298 | 0.6604 | 0.7723 | 0.9987 | 4 |
| Abt–Buy | Rationale 3-shot | 3000 | 0.9657 | 0.8972 | 0.7906 | 0.8405 | 0.9977 | 7 |
| Abt–Buy | Rationale 4-shot | 3000 | 0.9597 | 0.8859 | 0.7236 | 0.7966 | 0.9993 | 2 |
| Amazon–Google | Direct 0-shot | 1000 | 0.9120 | 0.6167 | 0.3627 | 0.4568 | 1.0000 | 0 |
| Amazon–Google | Direct 1-shot | 3000 | 0.8880 | 0.4698 | 0.7614 | 0.5810 | 1.0000 | 0 |
| Amazon–Google | Direct 2-shot | 3000 | 0.9083 | 0.5326 | 0.8268 | 0.6479 | 1.0000 | 0 |
| Amazon–Google | Direct 3-shot | 3000 | 0.8853 | 0.4682 | 0.9150 | 0.6195 | 1.0000 | 0 |
| Amazon–Google | Direct 4-shot | 3000 | 0.9030 | 0.5145 | 0.8693 | 0.6464 | 1.0000 | 0 |
| Amazon–Google | Rationale 0-shot | 1000 | 0.9160 | 0.5662 | 0.7549 | 0.6471 | 1.0000 | 0 |
| Amazon–Google | Rationale 1-shot | 3000 | 0.8533 | 0.3919 | 0.7941 | 0.5248 | 1.0000 | 0 |
| Amazon–Google | Rationale 2-shot | 3000 | 0.9060 | 0.5260 | 0.7941 | 0.6328 | 1.0000 | 0 |
| Amazon–Google | Rationale 3-shot | 3000 | 0.8813 | 0.4582 | 0.8954 | 0.6062 | 1.0000 | 0 |
| Amazon–Google | Rationale 4-shot | 3000 | 0.8947 | 0.4904 | 0.8366 | 0.6184 | 1.0000 | 0 |
| Walmart–Amazon | Direct 0-shot | 1000 | 0.9430 | 0.9302 | 0.4255 | 0.5839 | 1.0000 | 0 |
| Walmart–Amazon | Direct 1-shot | 3000 | 0.9530 | 0.7621 | 0.7270 | 0.7441 | 1.0000 | 0 |
| Walmart–Amazon | Direct 2-shot | 3000 | 0.9593 | 0.8333 | 0.7092 | 0.7663 | 1.0000 | 0 |
| Walmart–Amazon | Direct 3-shot | 3000 | 0.9507 | 0.7055 | 0.8156 | 0.7566 | 1.0000 | 0 |
| Walmart–Amazon | Direct 4-shot | 3000 | 0.9487 | 0.6939 | 0.8121 | 0.7484 | 1.0000 | 0 |
| Walmart–Amazon | Rationale 0-shot | 1000 | 0.9610 | 0.8161 | 0.7553 | 0.7845 | 1.0000 | 0 |
| Walmart–Amazon | Rationale 1-shot | 3000 | 0.9443 | 0.6727 | 0.7943 | 0.7285 | 1.0000 | 0 |
| Walmart–Amazon | Rationale 2-shot | 3000 | 0.9510 | 0.7778 | 0.6702 | 0.7200 | 1.0000 | 0 |
| Walmart–Amazon | Rationale 3-shot | 3000 | 0.9370 | 0.6306 | 0.8050 | 0.7072 | 0.9997 | 1 |
| Walmart–Amazon | Rationale 4-shot | 3000 | 0.9450 | 0.6997 | 0.7270 | 0.7130 | 1.0000 | 0 |
| WDC Products | Direct 0-shot | 1000 | 0.9240 | 0.8317 | 0.5874 | 0.6885 | 1.0000 | 0 |
| WDC Products | Direct 1-shot | 3000 | 0.9503 | 0.9242 | 0.7110 | 0.8037 | 1.0000 | 0 |
| WDC Products | Direct 2-shot | 3000 | 0.9473 | 0.8144 | 0.8182 | 0.8163 | 1.0000 | 0 |
| WDC Products | Direct 3-shot | 3000 | 0.9603 | 0.9036 | 0.8089 | 0.8536 | 1.0000 | 0 |
| WDC Products | Direct 4-shot | 3000 | 0.9570 | 0.8261 | 0.8858 | 0.8549 | 1.0000 | 0 |
| WDC Products | Rationale 0-shot | 1000 | 0.9440 | 0.8188 | 0.8014 | 0.8100 | 0.9970 | 3 |
| WDC Products | Rationale 1-shot | 3000 | 0.9637 | 0.8943 | 0.8505 | 0.8719 | 0.9993 | 2 |
| WDC Products | Rationale 2-shot | 3000 | 0.9553 | 0.8228 | 0.8785 | 0.8497 | 0.9997 | 1 |
| WDC Products | Rationale 3-shot | 3000 | 0.9667 | 0.8986 | 0.8671 | 0.8826 | 0.9997 | 1 |
| WDC Products | Rationale 4-shot | 3000 | 0.9613 | 0.8366 | 0.9089 | 0.8712 | 0.9997 | 1 |
| Link-Lives | Direct 0-shot | 1000 | 0.8330 | 1.0000 | 0.0060 | 0.0118 | 1.0000 | 0 |
| Link-Lives | Direct 1-shot | 3000 | 0.8483 | 0.9804 | 0.0992 | 0.1802 | 1.0000 | 0 |
| Link-Lives | Direct 2-shot | 3000 | 0.8470 | 0.9787 | 0.0913 | 0.1670 | 1.0000 | 0 |
| Link-Lives | Direct 3-shot | 3000 | 0.8667 | 0.9727 | 0.2123 | 0.3485 | 1.0000 | 0 |
| Link-Lives | Direct 4-shot | 3000 | 0.8680 | 0.9655 | 0.2222 | 0.3613 | 1.0000 | 0 |
| Link-Lives | Rationale 0-shot | 1000 | 0.8530 | 1.0000 | 0.1257 | 0.2234 | 0.9990 | 1 |
| Link-Lives | Rationale 1-shot | 3000 | 0.8633 | 0.9796 | 0.1905 | 0.3189 | 1.0000 | 0 |
| Link-Lives | Rationale 2-shot | 3000 | 0.8490 | 1.0000 | 0.1016 | 0.1844 | 0.9993 | 2 |
| Link-Lives | Rationale 3-shot | 3000 | 0.8650 | 0.9902 | 0.2012 | 0.3344 | 0.9990 | 3 |
| Link-Lives | Rationale 4-shot | 3000 | 0.8620 | 0.9891 | 0.1820 | 0.3074 | 0.9987 | 4 |

## Ministral 3 8B — validation

| Dataset | Prompt | N | Accuracy | Precision | Recall | F1 | Parse | Invalid |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Abt–Buy | Direct 0-shot | 1000 | 0.9400 | 0.9444 | 0.4722 | 0.6296 | 1.0000 | 0 |
| Abt–Buy | Direct 1-shot | 3000 | 0.9047 | 0.5344 | 0.9105 | 0.6735 | 1.0000 | 0 |
| Abt–Buy | Direct 2-shot | 3000 | 0.8897 | 0.4945 | 0.9630 | 0.6534 | 1.0000 | 0 |
| Abt–Buy | Direct 3-shot | 3000 | 0.9247 | 0.5992 | 0.9136 | 0.7237 | 1.0000 | 0 |
| Abt–Buy | Direct 4-shot | 3000 | 0.9153 | 0.5643 | 0.9475 | 0.7074 | 1.0000 | 0 |
| Abt–Buy | Rationale 0-shot | 1000 | 0.9160 | 0.5659 | 0.9537 | 0.7103 | 1.0000 | 0 |
| Abt–Buy | Rationale 1-shot | 3000 | 0.9130 | 0.5624 | 0.8793 | 0.6860 | 0.9997 | 1 |
| Abt–Buy | Rationale 2-shot | 3000 | 0.8697 | 0.4598 | 0.9567 | 0.6211 | 0.9953 | 14 |
| Abt–Buy | Rationale 3-shot | 3000 | 0.9137 | 0.5776 | 0.9130 | 0.7076 | 0.9947 | 16 |
| Abt–Buy | Rationale 4-shot | 3000 | 0.9103 | 0.5589 | 0.9344 | 0.6994 | 0.9960 | 12 |
| Amazon–Google | Direct 0-shot | 1000 | 0.9020 | 0.5714 | 0.1569 | 0.2462 | 1.0000 | 0 |
| Amazon–Google | Direct 1-shot | 3000 | 0.8380 | 0.3520 | 0.6993 | 0.4683 | 1.0000 | 0 |
| Amazon–Google | Direct 2-shot | 3000 | 0.7307 | 0.2641 | 0.9183 | 0.4102 | 1.0000 | 0 |
| Amazon–Google | Direct 3-shot | 3000 | 0.7560 | 0.2870 | 0.9379 | 0.4395 | 1.0000 | 0 |
| Amazon–Google | Direct 4-shot | 3000 | 0.7660 | 0.2959 | 0.9379 | 0.4498 | 1.0000 | 0 |
| Amazon–Google | Rationale 0-shot | 1000 | 0.7990 | 0.3160 | 0.8333 | 0.4582 | 1.0000 | 0 |
| Amazon–Google | Rationale 1-shot | 3000 | 0.8540 | 0.3813 | 0.6928 | 0.4919 | 1.0000 | 0 |
| Amazon–Google | Rationale 2-shot | 3000 | 0.7937 | 0.3178 | 0.8889 | 0.4682 | 0.9997 | 1 |
| Amazon–Google | Rationale 3-shot | 3000 | 0.7883 | 0.3183 | 0.9248 | 0.4736 | 0.9980 | 6 |
| Amazon–Google | Rationale 4-shot | 3000 | 0.7860 | 0.3185 | 0.9379 | 0.4756 | 0.9970 | 9 |
| Walmart–Amazon | Direct 0-shot | 1000 | 0.9200 | 0.7500 | 0.2234 | 0.3443 | 1.0000 | 0 |
| Walmart–Amazon | Direct 1-shot | 3000 | 0.8650 | 0.3820 | 0.7057 | 0.4956 | 1.0000 | 0 |
| Walmart–Amazon | Direct 2-shot | 3000 | 0.7810 | 0.2699 | 0.7801 | 0.4011 | 1.0000 | 0 |
| Walmart–Amazon | Direct 3-shot | 3000 | 0.7903 | 0.2839 | 0.8085 | 0.4203 | 1.0000 | 0 |
| Walmart–Amazon | Direct 4-shot | 3000 | 0.7103 | 0.2239 | 0.8440 | 0.3539 | 1.0000 | 0 |
| Walmart–Amazon | Rationale 0-shot | 1000 | 0.8610 | 0.3764 | 0.7128 | 0.4926 | 0.9990 | 1 |
| Walmart–Amazon | Rationale 1-shot | 3000 | 0.8833 | 0.4295 | 0.7340 | 0.5419 | 1.0000 | 0 |
| Walmart–Amazon | Rationale 2-shot | 3000 | 0.7783 | 0.2715 | 0.7943 | 0.4047 | 0.9980 | 6 |
| Walmart–Amazon | Rationale 3-shot | 3000 | 0.7630 | 0.2694 | 0.8790 | 0.4124 | 0.9977 | 7 |
| Walmart–Amazon | Rationale 4-shot | 3000 | 0.7173 | 0.2377 | 0.8932 | 0.3755 | 0.9957 | 13 |
| WDC Products | Direct 0-shot | 1000 | 0.8880 | 0.7067 | 0.3706 | 0.4862 | 1.0000 | 0 |
| WDC Products | Direct 1-shot | 3000 | 0.9327 | 0.8721 | 0.6200 | 0.7248 | 1.0000 | 0 |
| WDC Products | Direct 2-shot | 3000 | 0.8243 | 0.4417 | 0.8648 | 0.5847 | 1.0000 | 0 |
| WDC Products | Direct 3-shot | 3000 | 0.9200 | 0.7015 | 0.7669 | 0.7327 | 1.0000 | 0 |
| WDC Products | Direct 4-shot | 3000 | 0.8747 | 0.5388 | 0.8578 | 0.6619 | 1.0000 | 0 |
| WDC Products | Rationale 0-shot | 1000 | 0.8590 | 0.5134 | 0.8042 | 0.6267 | 0.9960 | 4 |
| WDC Products | Rationale 1-shot | 3000 | 0.9230 | 0.7703 | 0.6643 | 0.7134 | 0.9993 | 2 |
| WDC Products | Rationale 2-shot | 3000 | 0.7973 | 0.4078 | 0.8806 | 0.5574 | 0.9963 | 11 |
| WDC Products | Rationale 3-shot | 3000 | 0.8883 | 0.5825 | 0.8122 | 0.6784 | 0.9977 | 7 |
| WDC Products | Rationale 4-shot | 3000 | 0.8307 | 0.4598 | 0.8836 | 0.6049 | 0.9927 | 22 |
| Link-Lives | Direct 0-shot | 1000 | 0.8740 | 0.8500 | 0.3036 | 0.4474 | 1.0000 | 0 |
| Link-Lives | Direct 1-shot | 3000 | 0.9007 | 0.8199 | 0.5238 | 0.6392 | 1.0000 | 0 |
| Link-Lives | Direct 2-shot | 3000 | 0.9157 | 0.7687 | 0.7123 | 0.7394 | 1.0000 | 0 |
| Link-Lives | Direct 3-shot | 3000 | 0.9203 | 0.8338 | 0.6567 | 0.7347 | 1.0000 | 0 |
| Link-Lives | Direct 4-shot | 3000 | 0.9190 | 0.8071 | 0.6806 | 0.7384 | 1.0000 | 0 |
| Link-Lives | Rationale 0-shot | 1000 | 0.7730 | 0.4212 | 0.9226 | 0.5784 | 0.9990 | 1 |
| Link-Lives | Rationale 1-shot | 3000 | 0.8937 | 0.7866 | 0.5129 | 0.6209 | 0.9987 | 4 |
| Link-Lives | Rationale 2-shot | 3000 | 0.9023 | 0.7172 | 0.7044 | 0.7107 | 0.9987 | 4 |
| Link-Lives | Rationale 3-shot | 3000 | 0.9147 | 0.8073 | 0.6581 | 0.7251 | 0.9983 | 5 |
| Link-Lives | Rationale 4-shot | 3000 | 0.9163 | 0.7668 | 0.7256 | 0.7457 | 0.9993 | 2 |

## Phi-4-mini — validation

| Dataset | Prompt | N | Accuracy | Precision | Recall | F1 | Parse | Invalid |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Abt–Buy | Direct 0-shot | 1000 | 0.9250 | 0.8367 | 0.3796 | 0.5223 | 1.0000 | 0 |
| Abt–Buy | Direct 1-shot | 3000 | 0.9203 | 0.5986 | 0.7963 | 0.6834 | 1.0000 | 0 |
| Abt–Buy | Direct 2-shot | 3000 | 0.9400 | 0.7647 | 0.6420 | 0.6980 | 1.0000 | 0 |
| Abt–Buy | Direct 3-shot | 3000 | 0.9230 | 0.6110 | 0.7901 | 0.6891 | 1.0000 | 0 |
| Abt–Buy | Direct 4-shot | 3000 | 0.9360 | 0.6844 | 0.7562 | 0.7185 | 1.0000 | 0 |
| Abt–Buy | Rationale 0-shot | 1000 | 0.9140 | 0.8478 | 0.3714 | 0.5166 | 0.9870 | 13 |
| Abt–Buy | Rationale 1-shot | 3000 | 0.9283 | 0.6493 | 0.7315 | 0.6880 | 1.0000 | 0 |
| Abt–Buy | Rationale 2-shot | 3000 | 0.9380 | 0.7464 | 0.6451 | 0.6921 | 1.0000 | 0 |
| Abt–Buy | Rationale 3-shot | 3000 | 0.9303 | 0.6533 | 0.7562 | 0.7010 | 1.0000 | 0 |
| Abt–Buy | Rationale 4-shot | 3000 | 0.9353 | 0.6847 | 0.7438 | 0.7130 | 1.0000 | 0 |
| Amazon–Google | Direct 0-shot | 1000 | 0.9030 | 0.6190 | 0.1275 | 0.2114 | 1.0000 | 0 |
| Amazon–Google | Direct 1-shot | 3000 | 0.8973 | 0.4968 | 0.5098 | 0.5032 | 1.0000 | 0 |
| Amazon–Google | Direct 2-shot | 3000 | 0.8890 | 0.4630 | 0.5523 | 0.5037 | 1.0000 | 0 |
| Amazon–Google | Direct 3-shot | 3000 | 0.8927 | 0.4791 | 0.5980 | 0.5320 | 1.0000 | 0 |
| Amazon–Google | Direct 4-shot | 3000 | 0.8713 | 0.4203 | 0.6895 | 0.5223 | 1.0000 | 0 |
| Amazon–Google | Rationale 0-shot | 1000 | 0.8870 | 0.3902 | 0.1569 | 0.2238 | 0.9980 | 2 |
| Amazon–Google | Rationale 1-shot | 3000 | 0.8973 | 0.4963 | 0.4412 | 0.4671 | 1.0000 | 0 |
| Amazon–Google | Rationale 2-shot | 3000 | 0.8717 | 0.4128 | 0.6111 | 0.4928 | 1.0000 | 0 |
| Amazon–Google | Rationale 3-shot | 3000 | 0.8937 | 0.4812 | 0.5425 | 0.5100 | 1.0000 | 0 |
| Amazon–Google | Rationale 4-shot | 3000 | 0.8677 | 0.4140 | 0.7157 | 0.5246 | 1.0000 | 0 |
| Walmart–Amazon | Direct 0-shot | 1000 | 0.7630 | 0.1983 | 0.5000 | 0.2840 | 1.0000 | 0 |
| Walmart–Amazon | Direct 1-shot | 3000 | 0.7437 | 0.1749 | 0.4645 | 0.2541 | 1.0000 | 0 |
| Walmart–Amazon | Direct 2-shot | 3000 | 0.7593 | 0.1812 | 0.4433 | 0.2572 | 1.0000 | 0 |
| Walmart–Amazon | Direct 3-shot | 3000 | 0.7303 | 0.1735 | 0.4965 | 0.2571 | 1.0000 | 0 |
| Walmart–Amazon | Direct 4-shot | 3000 | 0.7333 | 0.1857 | 0.5426 | 0.2767 | 1.0000 | 0 |
| Walmart–Amazon | Rationale 0-shot | 1000 | 0.8980 | 0.4231 | 0.2340 | 0.3014 | 1.0000 | 0 |
| Walmart–Amazon | Rationale 1-shot | 3000 | 0.7807 | 0.1877 | 0.4007 | 0.2557 | 1.0000 | 0 |
| Walmart–Amazon | Rationale 2-shot | 3000 | 0.7623 | 0.1881 | 0.4610 | 0.2672 | 1.0000 | 0 |
| Walmart–Amazon | Rationale 3-shot | 3000 | 0.7603 | 0.1909 | 0.4787 | 0.2730 | 1.0000 | 0 |
| Walmart–Amazon | Rationale 4-shot | 3000 | 0.7177 | 0.1808 | 0.5674 | 0.2742 | 1.0000 | 0 |
| WDC Products | Direct 0-shot | 1000 | 0.8820 | 0.6506 | 0.3776 | 0.4779 | 1.0000 | 0 |
| WDC Products | Direct 1-shot | 3000 | 0.8960 | 0.6512 | 0.5874 | 0.6176 | 1.0000 | 0 |
| WDC Products | Direct 2-shot | 3000 | 0.8573 | 0.5009 | 0.6737 | 0.5746 | 1.0000 | 0 |
| WDC Products | Direct 3-shot | 3000 | 0.8700 | 0.5455 | 0.5455 | 0.5455 | 1.0000 | 0 |
| WDC Products | Direct 4-shot | 3000 | 0.8490 | 0.4799 | 0.6690 | 0.5589 | 1.0000 | 0 |
| WDC Products | Rationale 0-shot | 1000 | 0.8830 | 0.7500 | 0.3566 | 0.4834 | 0.9920 | 8 |
| WDC Products | Rationale 1-shot | 3000 | 0.8910 | 0.6656 | 0.4779 | 0.5563 | 1.0000 | 0 |
| WDC Products | Rationale 2-shot | 3000 | 0.8353 | 0.4513 | 0.7016 | 0.5493 | 1.0000 | 0 |
| WDC Products | Rationale 3-shot | 3000 | 0.8623 | 0.5173 | 0.5571 | 0.5365 | 1.0000 | 0 |
| WDC Products | Rationale 4-shot | 3000 | 0.8423 | 0.4650 | 0.6807 | 0.5525 | 1.0000 | 0 |
| Link-Lives | Direct 0-shot | 1000 | 0.8460 | 1.0000 | 0.0833 | 0.1538 | 1.0000 | 0 |
| Link-Lives | Direct 1-shot | 3000 | 0.8493 | 0.6884 | 0.1885 | 0.2960 | 1.0000 | 0 |
| Link-Lives | Direct 2-shot | 3000 | 0.8407 | 0.5637 | 0.2282 | 0.3249 | 1.0000 | 0 |
| Link-Lives | Direct 3-shot | 3000 | 0.8067 | 0.3995 | 0.2996 | 0.3424 | 1.0000 | 0 |
| Link-Lives | Direct 4-shot | 3000 | 0.7277 | 0.2954 | 0.4484 | 0.3562 | 1.0000 | 0 |
| Link-Lives | Rationale 0-shot | 1000 | 0.8650 | 0.9286 | 0.2335 | 0.3732 | 0.9960 | 4 |
| Link-Lives | Rationale 1-shot | 3000 | 0.8570 | 0.7451 | 0.2262 | 0.3470 | 1.0000 | 0 |
| Link-Lives | Rationale 2-shot | 3000 | 0.8263 | 0.4732 | 0.2976 | 0.3654 | 1.0000 | 0 |
| Link-Lives | Rationale 3-shot | 3000 | 0.7953 | 0.3750 | 0.3274 | 0.3496 | 1.0000 | 0 |
| Link-Lives | Rationale 4-shot | 3000 | 0.7397 | 0.3175 | 0.4782 | 0.3816 | 1.0000 | 0 |

## Qwen3-1.7B — validation

| Dataset | Prompt | N | Accuracy | Precision | Recall | F1 | Parse | Invalid |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Abt–Buy | Direct 0-shot | 1000 | 0.8970 | 1.0000 | 0.0463 | 0.0885 | 1.0000 | 0 |
| Abt–Buy | Direct 1-shot | 3000 | 0.9043 | 0.8136 | 0.1481 | 0.2507 | 1.0000 | 0 |
| Abt–Buy | Direct 2-shot | 3000 | 0.9103 | 0.6368 | 0.3951 | 0.4876 | 1.0000 | 0 |
| Abt–Buy | Direct 3-shot | 3000 | 0.9133 | 0.6111 | 0.5432 | 0.5752 | 1.0000 | 0 |
| Abt–Buy | Direct 4-shot | 3000 | 0.9123 | 0.6332 | 0.4475 | 0.5244 | 1.0000 | 0 |
| Abt–Buy | Rationale 0-shot | 1000 | 0.8240 | 0.7143 | 0.0602 | 0.1111 | 0.9040 | 96 |
| Abt–Buy | Rationale 1-shot | 3000 | 0.8983 | 0.9524 | 0.0617 | 0.1159 | 1.0000 | 0 |
| Abt–Buy | Rationale 2-shot | 3000 | 0.9060 | 0.8889 | 0.1481 | 0.2540 | 1.0000 | 0 |
| Abt–Buy | Rationale 3-shot | 3000 | 0.9127 | 0.9079 | 0.2130 | 0.3450 | 1.0000 | 0 |
| Abt–Buy | Rationale 4-shot | 3000 | 0.9117 | 0.8471 | 0.2222 | 0.3521 | 1.0000 | 0 |
| Amazon–Google | Direct 0-shot | 1000 | 0.8980 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0 |
| Amazon–Google | Direct 1-shot | 3000 | 0.8990 | 1.0000 | 0.0098 | 0.0194 | 1.0000 | 0 |
| Amazon–Google | Direct 2-shot | 3000 | 0.9003 | 0.6842 | 0.0425 | 0.0800 | 1.0000 | 0 |
| Amazon–Google | Direct 3-shot | 3000 | 0.9017 | 0.6897 | 0.0654 | 0.1194 | 1.0000 | 0 |
| Amazon–Google | Direct 4-shot | 3000 | 0.8997 | 0.6667 | 0.0327 | 0.0623 | 1.0000 | 0 |
| Amazon–Google | Rationale 0-shot | 1000 | 0.5260 | 0.0000 | 0.0000 | 0.0000 | 0.5920 | 408 |
| Amazon–Google | Rationale 1-shot | 3000 | 0.8753 | 0.0000 | 0.0000 | 0.0000 | 0.9737 | 79 |
| Amazon–Google | Rationale 2-shot | 3000 | 0.8980 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0 |
| Amazon–Google | Rationale 3-shot | 3000 | 0.8973 | 0.0000 | 0.0000 | 0.0000 | 0.9983 | 5 |
| Amazon–Google | Rationale 4-shot | 3000 | 0.8980 | 0.0000 | 0.0000 | 0.0000 | 0.9997 | 1 |
| Walmart–Amazon | Direct 0-shot | 1000 | 0.9080 | 1.0000 | 0.0213 | 0.0417 | 1.0000 | 0 |
| Walmart–Amazon | Direct 1-shot | 3000 | 0.8990 | 0.2439 | 0.0355 | 0.0619 | 1.0000 | 0 |
| Walmart–Amazon | Direct 2-shot | 3000 | 0.8227 | 0.1408 | 0.1738 | 0.1556 | 1.0000 | 0 |
| Walmart–Amazon | Direct 3-shot | 3000 | 0.7963 | 0.1416 | 0.2305 | 0.1754 | 1.0000 | 0 |
| Walmart–Amazon | Direct 4-shot | 3000 | 0.8280 | 0.1100 | 0.1170 | 0.1134 | 1.0000 | 0 |
| Walmart–Amazon | Rationale 0-shot | 1000 | 0.8620 | 0.0000 | 0.0000 | 0.0000 | 0.9520 | 48 |
| Walmart–Amazon | Rationale 1-shot | 3000 | 0.9063 | 0.7500 | 0.0106 | 0.0210 | 0.9997 | 1 |
| Walmart–Amazon | Rationale 2-shot | 3000 | 0.9037 | 0.3600 | 0.0319 | 0.0586 | 1.0000 | 0 |
| Walmart–Amazon | Rationale 3-shot | 3000 | 0.9047 | 0.4167 | 0.0355 | 0.0654 | 1.0000 | 0 |
| Walmart–Amazon | Rationale 4-shot | 3000 | 0.9020 | 0.3000 | 0.0319 | 0.0577 | 1.0000 | 0 |
| WDC Products | Direct 0-shot | 1000 | 0.8680 | 1.0000 | 0.0769 | 0.1429 | 1.0000 | 0 |
| WDC Products | Direct 1-shot | 3000 | 0.8653 | 0.8571 | 0.0699 | 0.1293 | 1.0000 | 0 |
| WDC Products | Direct 2-shot | 3000 | 0.8717 | 0.5701 | 0.4172 | 0.4818 | 1.0000 | 0 |
| WDC Products | Direct 3-shot | 3000 | 0.8577 | 0.5030 | 0.3916 | 0.4404 | 1.0000 | 0 |
| WDC Products | Direct 4-shot | 3000 | 0.8560 | 0.4971 | 0.5944 | 0.5414 | 1.0000 | 0 |
| WDC Products | Rationale 0-shot | 1000 | 0.7380 | 0.8333 | 0.0505 | 0.0952 | 0.8330 | 167 |
| WDC Products | Rationale 1-shot | 3000 | 0.8077 | 0.8750 | 0.0354 | 0.0680 | 0.9357 | 193 |
| WDC Products | Rationale 2-shot | 3000 | 0.8653 | 0.8137 | 0.1976 | 0.3180 | 0.9840 | 48 |
| WDC Products | Rationale 3-shot | 3000 | 0.8763 | 0.8113 | 0.2009 | 0.3221 | 0.9970 | 9 |
| WDC Products | Rationale 4-shot | 3000 | 0.8807 | 0.7294 | 0.2925 | 0.4175 | 0.9960 | 12 |
| Link-Lives | Direct 0-shot | 1000 | 0.8320 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0 |
| Link-Lives | Direct 1-shot | 3000 | 0.8340 | 1.0000 | 0.0119 | 0.0235 | 1.0000 | 0 |
| Link-Lives | Direct 2-shot | 3000 | 0.8500 | 0.8971 | 0.1210 | 0.2133 | 1.0000 | 0 |
| Link-Lives | Direct 3-shot | 3000 | 0.8630 | 0.7541 | 0.2738 | 0.4017 | 1.0000 | 0 |
| Link-Lives | Direct 4-shot | 3000 | 0.8530 | 0.7788 | 0.1746 | 0.2853 | 1.0000 | 0 |
| Link-Lives | Rationale 0-shot | 1000 | 0.1220 | 0.0000 | 0.0000 | 0.0000 | 0.1380 | 862 |
| Link-Lives | Rationale 1-shot | 3000 | 0.1243 | 0.0000 | 0.0000 | 0.0000 | 0.1467 | 2560 |
| Link-Lives | Rationale 2-shot | 3000 | 0.2310 | 0.7143 | 0.0413 | 0.0781 | 0.2703 | 2189 |
| Link-Lives | Rationale 3-shot | 3000 | 0.3513 | 1.0000 | 0.0825 | 0.1524 | 0.4107 | 1768 |
| Link-Lives | Rationale 4-shot | 3000 | 0.1413 | 0.9091 | 0.1429 | 0.2469 | 0.1617 | 2515 |

## Test audit

| Model | Dataset | Expected prompt rows | Stored test rows | Status |
|---|---|---:|---:|---|
| Qwen3.5 9B | Abt–Buy | 10 | 0 | Not run or not uploaded |
| Qwen3.5 9B | Amazon–Google | 10 | 0 | Not run or not uploaded |
| Qwen3.5 9B | Walmart–Amazon | 10 | 0 | Not run or not uploaded |
| Qwen3.5 9B | WDC Products | 10 | 0 | Not run or not uploaded |
| Qwen3.5 9B | Link-Lives | 10 | 0 | Not run or not uploaded |
| Ministral 3 8B | Abt–Buy | 10 | 0 | Not run or not uploaded |
| Ministral 3 8B | Amazon–Google | 10 | 0 | Not run or not uploaded |
| Ministral 3 8B | Walmart–Amazon | 10 | 0 | Not run or not uploaded |
| Ministral 3 8B | WDC Products | 10 | 0 | Not run or not uploaded |
| Ministral 3 8B | Link-Lives | 10 | 0 | Not run or not uploaded |
| Phi-4-mini | Abt–Buy | 10 | 0 | Not run or not uploaded |
| Phi-4-mini | Amazon–Google | 10 | 0 | Not run or not uploaded |
| Phi-4-mini | Walmart–Amazon | 10 | 0 | Not run or not uploaded |
| Phi-4-mini | WDC Products | 10 | 0 | Not run or not uploaded |
| Phi-4-mini | Link-Lives | 10 | 0 | Not run or not uploaded |
| Qwen3-1.7B | Abt–Buy | 10 | 0 | Not run or not uploaded |
| Qwen3-1.7B | Amazon–Google | 10 | 0 | Not run or not uploaded |
| Qwen3-1.7B | Walmart–Amazon | 10 | 0 | Not run or not uploaded |
| Qwen3-1.7B | WDC Products | 10 | 0 | Not run or not uploaded |
| Qwen3-1.7B | Link-Lives | 10 | 0 | Not run or not uploaded |
