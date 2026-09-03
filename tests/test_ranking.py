import pandas as pd

from paper1_hef.exp02 import _ranking_metrics


def test_ranking_ties_fall_back_to_retrieval_rank() -> None:
    frame = pd.DataFrame(
        {
            "query_id": ["q1", "q1", "q1", "q2", "q2"],
            "retrieval_rank": [2, 1, 3, 2, 1],
            "label": [1, 0, 0, 0, 1],
            "score": [0.5, 0.5, 0.1, 0.8, 0.8],
        }
    )
    metrics = _ranking_metrics(frame, "score")
    assert metrics["queries"] == 2
    assert metrics["mrr"] == 0.75
    assert metrics["hits_at_1"] == 0.5
    assert metrics["hits_at_100"] == 1.0
