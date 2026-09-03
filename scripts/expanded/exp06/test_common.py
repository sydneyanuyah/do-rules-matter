from exp06_common import CONDITIONS, selected_hybrid_features, selected_structured_features


PRODUCT = [
    "title_jaccard", "title_char_ratio", "description_jaccard", "all_token_jaccard",
    "price_similarity", "numeric_token_jaccard", "brand_exact", "manufacturer_exact",
    "currency_exact", "model_exact", "rule_score", "left_field_fraction",
    "right_field_fraction", "shared_field_fraction",
]


def test_nine_conditions_are_distinct_and_valid():
    assert len(CONDITIONS) == 9
    outputs = {}
    for condition in CONDITIONS:
        names, semantic = selected_structured_features(PRODUCT, condition)
        assert names
        outputs[condition] = (tuple(names), semantic)
    assert len(set(outputs.values())) == 9
    assert outputs["drop_semantic"][1] is False
    assert outputs["raw_field_features_only"][1] is False
    assert outputs["aggregate_rule_score_only"] == (("rule_score",), False)


def test_semantic_columns_are_removed_together():
    names = selected_hybrid_features(PRODUCT, "drop_semantic", ["embedding_score", "tuned_roberta_score"])
    assert "embedding_score" not in names and "tuned_roberta_score" not in names
    fullish = selected_hybrid_features(PRODUCT, "drop_lexical", ["embedding_score", "tuned_roberta_score"])
    assert "embedding_score" in fullish and "tuned_roberta_score" in fullish
