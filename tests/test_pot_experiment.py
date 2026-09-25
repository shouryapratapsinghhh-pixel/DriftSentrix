import copy

from anomaly.experiment import load_config, run_experiment


def test_pot_threshold_mode_runs_end_to_end():
    config = {
        "dataset": "synthetic",
        "root": None,
        "entities": "all",
        "synthetic_n_train": 500,
        "synthetic_n_test": 300,
        "synthetic_d": 3,
        "seeds": [0],
        "models": [{"name": "zscore", "params": {}}],
        "thresholds": ["pot"],
        "pot": [{"level": 0.95, "q": 0.01}],
        "output_dir": "reports/runs",
    }
    df = run_experiment(copy.deepcopy(config))
    assert len(df) == 1
    assert df.iloc[0]["threshold_mode"] == "pot_l0.95_q0.01"
    assert df.iloc[0]["point_f1"] >= 0.0  # ran without crashing, produced a valid metric


def test_smoke_config_includes_pot():
    config = load_config("configs/synthetic_smoke.yaml")
    assert "pot" in config["thresholds"]
