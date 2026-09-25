import numpy as np
import pytest

from anomaly.export import export_artifact
from anomaly.serve.state import load_state

pytestmark = pytest.mark.slow  # trains a real torch model

TINY_CONFIG = {
    "dataset": "synthetic",
    "root": None,
    "entities": "all",
    "synthetic_n_train": 200,
    "synthetic_n_test": 100,
    "synthetic_d": 3,
    "seeds": [0],
    "models": [
        {
            "name": "lstm_ae",
            "params": {
                "window": 8, "hidden": 8, "layers": 1, "latent": 4,
                "batch_size": 16, "max_epochs": 3, "patience": 2,
            },
        }
    ],
    "thresholds": ["oracle"],
    "output_dir": "reports/runs",
}


def test_oracle_threshold_mode_rejected():
    with pytest.raises(ValueError, match="oracle"):
        export_artifact(TINY_CONFIG, "lstm_ae", "oracle", "unused_out")


def test_unknown_model_name_rejected(tmp_path):
    with pytest.raises(ValueError, match="no model named"):
        export_artifact(TINY_CONFIG, "not_a_model", "pot", tmp_path / "out")


def test_export_then_load_state_round_trip(tmp_path):
    """Regression test: export_artifact writes a 'detector_type' key into
    config.json (for state.py to know which class to reconstruct) --
    load_state/Detector.load() must tolerate that extra key and not
    blow up passing it straight into the constructor.
    """
    out_dir = tmp_path / "artifact"
    export_artifact(TINY_CONFIG, "lstm_ae", "train_percentile", out_dir, percentile=99.0)

    state = load_state(out_dir)
    assert state.detector_type == "lstm_ae"
    assert state.n_features == 3
    assert state.threshold_mode == "train_percentile"

    # the loaded detector must actually work, not just construct
    x = np.random.default_rng(0).normal(size=(20, 3)).astype(np.float32)
    scores = state.detector.score(x)
    assert scores.shape == (20,)
    assert np.all(np.isfinite(scores))
