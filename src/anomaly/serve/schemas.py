from __future__ import annotations

from pydantic import BaseModel, Field


class ScoreRequest(BaseModel):
    values: list[list[float]] = Field(..., description="A batch series: (n_timesteps, n_features)")


class ScoreResponse(BaseModel):
    scores: list[float]
    alerts: list[bool]


class StreamPushRequest(BaseModel):
    x: list[float] = Field(..., description="One timestep's feature vector")


class DriftInfo(BaseModel):
    status: str  # "ok" | "warn" | "alert" | "unavailable"


class DriftFeatureReport(BaseModel):
    name: str
    psi: float
    ks: float
    ks_alert: bool
    psi_alert: bool


class DriftReport(BaseModel):
    """Full detail for GET /drift/{entity_id} -- the compact DriftInfo
    embedded in every /stream/{id}/push response is just this report's
    `status` field, to keep that hot path's payload small.
    """

    status: str  # "ok" | "warn" | "alert" | "unavailable" | "insufficient_data"
    n_current: int = 0
    n_required: int = 0
    features: list[DriftFeatureReport] = []
    top_drifted_features: list[str] = []


class StreamPushResponse(BaseModel):
    score: float | None
    alert: bool
    warmup: bool
    drift: DriftInfo


class ModelInfo(BaseModel):
    name: str
    window: int
    n_features: int
    threshold_mode: str
    threshold_value: float
    version: str = "v1"


class HealthResponse(BaseModel):
    status: str = "ok"


class MetricsResponse(BaseModel):
    requests_total: int
    alerts_total: int
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
