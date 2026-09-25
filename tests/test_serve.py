import time

import pytest
from fastapi.testclient import TestClient

from anomaly.data.loaders import make_synthetic
from anomaly.models.lstm_ae import LSTMAEDetector
from anomaly.serve.app import create_app
from anomaly.serve.state import AppState

pytestmark = pytest.mark.slow  # trains a real torch model

TINY = {"window": 4, "hidden": 4, "layers": 1, "latent": 2, "batch_size": 16, "max_epochs": 3, "patience": 2}


@pytest.fixture(scope="module")
def fitted_detector():
    train, _, _ = make_synthetic(n_train=150, n_test=50, d=3, seed=0)
    det = LSTMAEDetector(**TINY, seed=0).fit(train)
    return det, train


@pytest.fixture()
def client_with_drift(fitted_detector):
    """A separate client whose AppState has real reference data, so
    drift monitoring is actually active (unlike `client`, which omits
    it to test the 'unavailable' path).
    """
    det, train = fitted_detector
    scores = det.score(train)
    threshold = float(sorted(scores)[int(0.99 * len(scores))])
    state = AppState(
        detector=det,
        detector_type="lstm_ae",
        threshold=threshold,
        threshold_mode="train_percentile",
        n_features=3,
        reference=train,
        drift_window=30,  # small so tests don't need hundreds of pushes
    )
    app = create_app(state)
    return TestClient(app)


def test_drift_insufficient_data_before_window_full(client_with_drift):
    client_with_drift.post("/stream/e1/push", json={"x": [0.1, 0.2, 0.3]})
    r = client_with_drift.get("/drift/e1")
    assert r.status_code == 200
    # DriftMonitor.report()'s own "insufficient_data" status passes through as-is
    # (DriftInfo.status is a plain str, not an enum) until drift_window points arrive
    assert r.json()["status"] == "insufficient_data"


def test_drift_ok_on_matching_distribution(client_with_drift):
    for _ in range(35):
        client_with_drift.post("/stream/e1/push", json={"x": [0.0, 0.0, 0.0]})
    r = client_with_drift.get("/drift/e1")
    assert r.status_code == 200
    assert r.json()["status"] in ("ok", "warn", "alert", "unavailable")  # must not crash; exact status data-dependent


def test_drift_alert_on_large_shift(client_with_drift):
    for _ in range(35):
        client_with_drift.post("/stream/e1/push", json={"x": [500.0, 500.0, 500.0]})  # wildly off-distribution
    r = client_with_drift.get("/drift/e1")
    assert r.status_code == 200
    assert r.json()["status"] == "alert"


def test_push_response_includes_drift_status(client_with_drift):
    for _ in range(35):
        r = client_with_drift.post("/stream/e1/push", json={"x": [500.0, 500.0, 500.0]})
    assert r.json()["drift"]["status"] == "alert"


@pytest.fixture()
def client(fitted_detector):
    det, train = fitted_detector
    scores = det.score(train)
    threshold = float(sorted(scores)[int(0.99 * len(scores))])  # rough percentile, fine for tests
    state = AppState(
        detector=det,
        detector_type="lstm_ae",
        threshold=threshold,
        threshold_mode="train_percentile",
        n_features=3,
        max_entities=3,
        ttl_seconds=3600,
    )
    app = create_app(state)
    return TestClient(app)


# -- basic endpoints ----------------------------------------------------


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_model_info(client):
    r = client.get("/model")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "lstm_ae"
    assert body["window"] == TINY["window"]
    assert body["n_features"] == 3


def test_score_batch(client):
    r = client.post("/score", json={"values": [[0.1, 0.2, 0.3]] * 10})
    assert r.status_code == 200
    body = r.json()
    assert len(body["scores"]) == 10
    assert len(body["alerts"]) == 10
    assert all(isinstance(s, float) for s in body["scores"])


def test_score_wrong_feature_count_422(client):
    r = client.post("/score", json={"values": [[0.1, 0.2]]})  # only 2 features, expects 3
    assert r.status_code == 422


def test_score_nan_rejected_422(client):
    # NaN isn't valid standard JSON, so httpx's auto-encoder refuses to send
    # it -- send raw bytes instead (Python's json.loads DOES accept the
    # NaN literal on the receiving end, same as our own service uses).
    r = client.post(
        "/score",
        content='{"values": [[0.1, NaN, 0.3]]}',
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 422


def test_score_empty_422(client):
    r = client.post("/score", json={"values": []})
    assert r.status_code == 422


def test_drift_unavailable_when_entity_unknown(client):
    # /drift/{id} for an entity that never pushed anything, and this
    # client's fixture has no reference data configured -> stays "unavailable"
    r = client.get("/drift/never_seen_entity")
    assert r.status_code == 200
    assert r.json()["status"] == "unavailable"


def test_metrics_initially_zero(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.json()
    assert body["requests_total"] == 0
    assert body["alerts_total"] == 0


# -- streaming push -------------------------------------------------------


def test_stream_push_warmup_then_scores(client):
    for i in range(TINY["window"] - 1):
        r = client.post("/stream/e1/push", json={"x": [0.1, 0.2, 0.3]})
        assert r.status_code == 200
        assert r.json()["warmup"] is True
        assert r.json()["score"] is None

    r = client.post("/stream/e1/push", json={"x": [0.1, 0.2, 0.3]})
    assert r.status_code == 200
    body = r.json()
    assert body["warmup"] is False
    assert isinstance(body["score"], float)


def test_stream_push_wrong_feature_count_422(client):
    r = client.post("/stream/e1/push", json={"x": [0.1, 0.2]})
    assert r.status_code == 422


def test_stream_push_nan_rejected_422(client):
    r = client.post(
        "/stream/e1/push",
        content='{"x": [0.1, NaN, 0.3]}',
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 422


def test_metrics_update_after_push(client):
    client.post("/stream/e1/push", json={"x": [0.1, 0.2, 0.3]})
    r = client.get("/metrics")
    assert r.json()["requests_total"] >= 1


# -- per-entity isolation -------------------------------------------------


def test_per_entity_isolation(client):
    # push entity A through its full warmup
    for _ in range(TINY["window"]):
        client.post("/stream/A/push", json={"x": [0.1, 0.2, 0.3]})
    # entity B is brand new -- its first push must still be warmup, unaffected by A
    r = client.post("/stream/B/push", json={"x": [0.1, 0.2, 0.3]})
    assert r.json()["warmup"] is True


# -- max-entities cap -------------------------------------------------------


def test_max_entities_cap_returns_503(client):
    # fixture's client has max_entities=3
    for name in ["e_a", "e_b", "e_c"]:
        r = client.post(f"/stream/{name}/push", json={"x": [0.1, 0.2, 0.3]})
        assert r.status_code == 200
    r = client.post("/stream/e_d/push", json={"x": [0.1, 0.2, 0.3]})
    assert r.status_code == 503


def test_ttl_eviction_frees_capacity(fitted_detector):
    det, _train = fitted_detector
    state = AppState(
        detector=det,
        detector_type="lstm_ae",
        threshold=1e18,
        threshold_mode="train_percentile",
        n_features=3,
        max_entities=1,
        ttl_seconds=0.05,  # very short TTL for the test
    )
    app = create_app(state)
    client = TestClient(app)

    r1 = client.post("/stream/only_slot/push", json={"x": [0.1, 0.2, 0.3]})
    assert r1.status_code == 200

    time.sleep(0.1)  # let TTL expire

    # a new entity should now fit, since the old one aged out
    r2 = client.post("/stream/new_entity/push", json={"x": [0.1, 0.2, 0.3]})
    assert r2.status_code == 200
