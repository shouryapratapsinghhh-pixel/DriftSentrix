import copy
import time

import pandas as pd
import pytest

from anomaly.experiment import (
    load_config,
    run_experiment,
    validate_config,
)

SMOKE_CONFIG = "configs/synthetic_smoke.yaml"


def test_load_smoke_config():
    config = load_config(SMOKE_CONFIG)
    assert config["dataset"] == "synthetic"
    assert len(config["models"]) == 4


def test_validate_config_missing_key():
    bad = {"dataset": "synthetic"}
    with pytest.raises(ValueError, match="missing required keys"):
        validate_config(bad)


def test_validate_config_bad_dataset():
    bad = {
        "dataset": "not_a_real_dataset",
        "entities": "all",
        "seeds": [0],
        "models": [{"name": "zscore"}],
        "thresholds": ["oracle"],
        "output_dir": "x",
    }
    with pytest.raises(ValueError, match="dataset must be one of"):
        validate_config(bad)


def test_validate_config_unknown_model():
    bad = {
        "dataset": "synthetic",
        "entities": "all",
        "seeds": [0],
        "models": [{"name": "not_a_real_model"}],
        "thresholds": ["oracle"],
        "output_dir": "x",
    }
    with pytest.raises(ValueError, match="unknown model name"):
        validate_config(bad)


def test_validate_config_train_percentile_requires_list():
    bad = {
        "dataset": "synthetic",
        "entities": "all",
        "seeds": [0],
        "models": [{"name": "zscore"}],
        "thresholds": ["train_percentile"],
        "output_dir": "x",
    }
    with pytest.raises(ValueError, match="train_percentile"):
        validate_config(bad)


def test_experiment_runs_and_shape():
    config = load_config(SMOKE_CONFIG)
    df = run_experiment(copy.deepcopy(config))
    assert len(df) > 0
    assert {"entity", "seed", "model", "threshold_mode", "point_f1", "pa_f1", "event_f1"}.issubset(
        df.columns
    )
    # deterministic models (zscore, ewma, pca) should appear with only ONE seed each,
    # even though the config lists three seeds
    for name in ("zscore", "ewma", "pca"):
        assert df.loc[df["model"] == name, "seed"].nunique() == 1
    # stochastic model (isolation_forest) should appear with all three seeds
    assert df.loc[df["model"] == "isolation_forest", "seed"].nunique() == 3


def test_experiment_is_deterministic():
    config = load_config(SMOKE_CONFIG)
    df1 = run_experiment(copy.deepcopy(config))
    df2 = run_experiment(copy.deepcopy(config))
    pd.testing.assert_frame_equal(df1, df2)


def test_experiment_completes_quickly():
    config = load_config(SMOKE_CONFIG)
    start = time.time()
    run_experiment(copy.deepcopy(config))
    assert time.time() - start < 60
