import copy

import pandas as pd
import pytest

from anomaly.experiment import DETECTOR_REGISTRY, load_config, run_experiment
from anomaly.report import ablation_table


def test_deep_models_registered():
    assert "lstm_ae" in DETECTOR_REGISTRY
    assert "transformer_ae" in DETECTOR_REGISTRY


def test_ablation_config_loads_and_validates():
    config = load_config("configs/ablation.yaml")
    assert config["dataset"] == "synthetic"
    assert len(config["models"]) == 48  # 4 window x 3 hidden x 2 score_mode x 2 model types


@pytest.mark.slow  # trains two real torch models
def test_label_disambiguates_same_model_name_in_output():
    """Two config entries with the same underlying model name but
    different labels must NOT collide into one row group -- this is
    exactly what the ablation sweep depends on.
    """
    config = {
        "dataset": "synthetic",
        "root": None,
        "entities": "all",
        "synthetic_n_train": 200,
        "synthetic_n_test": 150,
        "synthetic_d": 3,
        "seeds": [0],
        "models": [
            {
                "name": "lstm_ae",
                "label": "lstm_ae_w8",
                "params": {
                    "window": 8, "hidden": 8, "layers": 1, "latent": 4,
                    "batch_size": 32, "max_epochs": 3, "patience": 2,
                },
            },
            {
                "name": "lstm_ae",
                "label": "lstm_ae_w16",
                "params": {
                    "window": 16, "hidden": 8, "layers": 1, "latent": 4,
                    "batch_size": 32, "max_epochs": 3, "patience": 2,
                },
            },
        ],
        "thresholds": ["oracle"],
        "output_dir": "reports/runs",
    }
    df = run_experiment(copy.deepcopy(config))
    assert set(df["model"]) == {"lstm_ae_w8", "lstm_ae_w16"}
    assert len(df) == 2  # one oracle row per label


def test_ablation_table_groups_by_label():
    df = pd.DataFrame(
        [
            {"entity": "synthetic", "seed": 0, "model": "lstm_ae_w8_h32_window_mean",
             "threshold_mode": "oracle", "point_f1": 0.5, "pa_f1": 0.6, "event_f1": 0.7,
             "point_precision": 0.5, "point_recall": 0.5, "pa_precision": 0.6, "pa_recall": 0.6,
             "event_precision": 0.7, "event_recall": 0.7, "mean_detection_delay": 1.0,
             "n_false_alarms": 0, "false_alarms_per_1000": 0.0},
            {"entity": "synthetic", "seed": 0, "model": "transformer_ae_w8_d32_window_mean",
             "threshold_mode": "oracle", "point_f1": 0.9, "pa_f1": 0.95, "event_f1": 0.95,
             "point_precision": 0.9, "point_recall": 0.9, "pa_precision": 0.95, "pa_recall": 0.95,
             "event_precision": 0.95, "event_recall": 0.95, "mean_detection_delay": 0.0,
             "n_false_alarms": 0, "false_alarms_per_1000": 0.0},
        ]
    )
    out = ablation_table(df)
    assert len(out) == 2  # two distinct settings, not collapsed into one
    assert set(out["model"]) == {"lstm_ae_w8_h32_window_mean", "transformer_ae_w8_d32_window_mean"}
