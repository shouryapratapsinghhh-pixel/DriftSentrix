"""FastAPI streaming anomaly-scoring service.

Endpoints: GET /health, GET /model, POST /score (batch), POST
/stream/{entity_id}/push (one timestep, stateful per entity), GET
/drift/{entity_id} (PSI/KS-based drift status, per entity), GET /metrics.

Model loading: set MODEL_DIR to an artifact directory produced by
export.py, then run:
    MODEL_DIR=artifacts/model_v1 uvicorn anomaly.serve.app:app

For tests, use create_app(state) directly with an in-memory AppState --
no environment variable, no file I/O needed (see tests/test_serve.py).
"""

from __future__ import annotations

import os
import time

import numpy as np
from fastapi import FastAPI, HTTPException

from anomaly.serve.schemas import (
    DriftFeatureReport,
    DriftInfo,
    DriftReport,
    HealthResponse,
    MetricsResponse,
    ModelInfo,
    ScoreRequest,
    ScoreResponse,
    StreamPushRequest,
    StreamPushResponse,
)
from anomaly.serve.state import AppState, load_state


def _validate_features(arr: np.ndarray, n_features: int, expected_ndim: int) -> None:
    if arr.ndim != expected_ndim or arr.shape[-1] != n_features:
        raise HTTPException(
            status_code=422,
            detail=f"expected {n_features} features per row, got shape {arr.shape}",
        )
    if not np.isfinite(arr).all():
        # documented policy: reject NaN/inf, never silently impute
        raise HTTPException(status_code=422, detail="input contains NaN/inf; rejected, not imputed")


def create_app(state: AppState) -> FastAPI:
    app = FastAPI(title="DriftSentrix")
    app.state.anomaly = state

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse()

    @app.get("/model", response_model=ModelInfo)
    def model_info() -> ModelInfo:
        return ModelInfo(
            name=state.detector_type,
            window=getattr(state.detector, "window", 1),
            n_features=state.n_features,
            threshold_mode=state.threshold_mode,
            threshold_value=state.threshold,
        )

    @app.post("/score", response_model=ScoreResponse)
    def score(req: ScoreRequest) -> ScoreResponse:
        arr = np.array(req.values, dtype=np.float64)
        if arr.size == 0:
            raise HTTPException(status_code=422, detail="values must be non-empty")
        _validate_features(arr, state.n_features, expected_ndim=2)

        scores = state.detector.score(arr.astype(np.float32))
        alerts = (scores >= state.threshold).tolist()
        return ScoreResponse(scores=[float(s) for s in scores], alerts=[bool(a) for a in alerts])

    @app.post("/stream/{entity_id}/push", response_model=StreamPushResponse)
    def stream_push(entity_id: str, req: StreamPushRequest) -> StreamPushResponse:
        start = time.perf_counter()
        x = np.array(req.x, dtype=np.float64)
        _validate_features(x, state.n_features, expected_ndim=1)

        try:
            scorer = state.get_scorer(entity_id)
        except RuntimeError as e:  # max entities reached
            raise HTTPException(status_code=503, detail=str(e)) from e

        result = scorer.push(x)
        latency_ms = (time.perf_counter() - start) * 1000
        state.record_request(latency_ms, alert=result.alert)

        drift_status = "unavailable"
        monitor = state.get_drift_monitor(entity_id)
        if monitor is not None:
            monitor.update(x)  # compact per-push signal only; full detail lives at GET /drift/{id}
            drift_status = monitor.report()["status"]

        return StreamPushResponse(
            score=result.score,
            alert=result.alert,
            warmup=result.warmup,
            drift=DriftInfo(status=drift_status),
        )

    @app.get("/drift/{entity_id}", response_model=DriftReport)
    def drift(entity_id: str) -> DriftReport:
        monitor = state.get_drift_monitor(entity_id)
        if monitor is None:
            return DriftReport(status="unavailable")
        report = monitor.report()
        return DriftReport(
            status=report["status"],
            n_current=report.get("n_current", 0),
            n_required=report.get("n_required", 0),
            features=[DriftFeatureReport(**f) for f in report.get("features", [])],
            top_drifted_features=report.get("top_drifted_features", []),
        )

    @app.get("/metrics", response_model=MetricsResponse)
    def metrics() -> MetricsResponse:
        p50, p95, p99 = state.latency_percentiles()
        return MetricsResponse(
            requests_total=state.requests_total,
            alerts_total=state.alerts_total,
            latency_p50_ms=p50,
            latency_p95_ms=p95,
            latency_p99_ms=p99,
        )

    return app


def _app_from_env() -> FastAPI:
    model_dir = os.environ.get("MODEL_DIR")
    if not model_dir:
        raise RuntimeError(
            "MODEL_DIR environment variable is not set. Run `python -m anomaly.export` "
            "first, then: MODEL_DIR=artifacts/model_v1 uvicorn anomaly.serve.app:app"
        )
    return create_app(load_state(model_dir))


# uvicorn anomaly.serve.app:app -- only built (and MODEL_DIR required) when actually
# run as a server; importing this module for its functions/classes (e.g. in tests)
# never triggers this.
app = None
if os.environ.get("MODEL_DIR"):
    app = _app_from_env()
