import pandas as pd

from paper1_hef.features import serialize, structured_features


def test_product_serialization_and_features() -> None:
    frame = pd.DataFrame(
        {
            "left_title": ["Acme Camera X100"],
            "right_title": ["ACME X100 Camera"],
            "left_brand": ["Acme"],
            "right_brand": ["acme"],
            "left_price": ["100"],
            "right_price": ["101"],
        }
    )
    assert "[COL] title [VAL] acme camera x100" in serialize(frame, "left").iloc[0]
    features = structured_features(frame)
    assert features.loc[0, "brand_exact"] == 1.0
    assert 0.0 < features.loc[0, "price_similarity"] <= 1.0
    assert features.notna().all().all()


def test_genealogy_schema_is_selected_from_name_fields() -> None:
    frame = pd.DataFrame(
        {
            "left_name": ["Ada Smith"],
            "right_name": ["Ada Smyth"],
            "left_birth_year": ["1880"],
            "right_birth_year": ["1881"],
        }
    )
    text = serialize(frame, "left").iloc[0]
    assert "[COL] name [VAL] ada smith" in text
    assert structured_features(frame).notna().all().all()
