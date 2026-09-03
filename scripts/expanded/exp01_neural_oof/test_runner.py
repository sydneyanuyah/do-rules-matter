from __future__ import annotations
import importlib.util, sys
from pathlib import Path
import numpy as np
import pandas as pd

PATH = Path(__file__).with_name("run_exp01_neural_oof_fusion.py")
SPEC = importlib.util.spec_from_file_location("neural_oof", PATH); assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC); sys.modules[SPEC.name] = RUNNER; SPEC.loader.exec_module(RUNNER)

def test_strict_component_folds_no_record_leakage_and_complete() -> None:
    frame = pd.DataFrame({
        "left_id": ["a","a","b","c","d","e","f","g","h"],
        "right_id": ["x","y","z","u","v","w","q","r","s"],
        "label": [1,0,0,1,0,1,0,1,0], "pair_id": [str(i) for i in range(9)]})
    plan = RUNNER.component_folds(frame, "deepmatcher")
    assert set(plan.assignment) == {0,1,2}
    assert plan.assignment[0] == plan.assignment[1]
    assert sum(x["holdout_rows"] for x in plan.report["folds"]) == len(frame)
    assert all(x["record_overlap"] == 0 for x in plan.report["folds"])
    again = RUNNER.component_folds(frame, "deepmatcher")
    assert np.array_equal(plan.assignment, again.assignment)

