# Architecture

```mermaid
flowchart TD
    subgraph Data
        A1[make_synthetic] --> W
        A2[load_smd] --> W
        A3[load_nasa] --> W
    end
    W[Windowing<br/>make_windows / window_scores_to_points] --> D

    subgraph Detectors
        D1[Random / Z-score / EWMA<br/>Isolation Forest / PCA]
        D2[LSTM-AE]
        D3[Transformer-AE]
    end
    W --> D1
    W --> D2
    W --> D3

    D1 --> T
    D2 --> T
    D3 --> T
    T[Threshold<br/>oracle / train_percentile / POT] --> S

    S[StreamingScorer<br/>ring buffer, causal] --> API

    subgraph Service [FastAPI]
        API[/score /model /health/]
        PUSH[/stream/{id}/push/]
        MET[/metrics/]
    end
    API --> PUSH
    PUSH --> DM

    DM[DriftMonitor<br/>PSI + KS per feature] -->|status: ok / warn / alert| PUSH
    PUSH --> MET
```

## Components

**Data (`data/loaders.py`, `data/windows.py`)** — three loaders share one
contract: `(train, test, labels)`, `train` assumed clean, `labels` binary
with `1 = anomalous`. `make_synthetic` needs no external files so tests and
CI run fully offline; `load_smd`/`load_nasa` read real datasets from a local
path and raise a clear error if the data isn't there. `make_windows` turns
`(n, d)` into `(n - window + 1, window, d)` sliding windows;
`window_scores_to_points` maps window scores back to one score per original
timestep, assigning each window's score to its **last** timestep — the
convention that makes every detector's `score()` causal (see below).

**Detectors (`models/`)** — every detector, baseline or deep, implements the
same `fit(train) -> self` / `score(x) -> ndarray` contract, so the experiment
harness, the streaming scorer, and the service never need to know which kind
of model they're holding. Baselines (`baselines.py`) are `window=1`
point-wise scorers except EWMA, whose score depends on the entire history via
a recurrent state (`reset_stream`/`update_stream`). `lstm_ae.py` and
`transformer_ae.py` share one training loop (`models/trainer.py`): chronological
train/val split (last 20% of windows, never shuffled across that boundary),
Adam with gradient clipping, early stopping with best-weights restore, full
seeding.

**Thresholds (`eval/thresholds.py`, `eval/pot.py`)** — `oracle` searches test
labels directly for the point-F1-maximizing cutoff: an upper bound, never a
deployable choice. `train_percentile` and `pot` (Peaks-Over-Threshold, a
Generalized Pareto Distribution fit to the tail of validation scores) are
both label-free and deployable; POT is the more principled of the two,
extrapolating into the tail via extreme value theory rather than just reading
off an empirical quantile.

**Streaming scorer (`streaming/scorer.py`)** — wraps any detector in a ring
buffer so it can be scored one timestep at a time, the way a real deployment
receives data. Returns no score until the buffer holds a full window
(warmup). Proven, per detector, to produce byte-for-byte the same scores as
batch `score()` from index `window - 1` onward
(`tests/test_streaming_scorer.py::test_streaming_equals_batch`) — this is the
single most load-bearing test in the repo, since everything downstream (the
service, latency numbers, drift monitoring on live pushes) assumes streaming
and batch scoring agree.

**Service (`serve/app.py`, `serve/state.py`)** — FastAPI. `AppState` holds one
`StreamingScorer` (and, if reference data was exported, one `DriftMonitor`)
per `entity_id`, with a max-entities cap and TTL-based eviction so a
long-running service doesn't accumulate unbounded per-entity state.
`GET /drift/{entity_id}` deliberately does **not** create an entity as a side
effect of a read — checking drift for an entity that never pushed anything is
meaningless, and a GET silently consuming a capacity slot would be a bad API.

**Drift monitor (`monitor/drift.py`)** — PSI and the KS statistic, both
implemented from scratch (scipy is used only in tests, to verify the KS
statistic matches `scipy.stats.ks_2samp`'s D-value, and never at
decision-time inside the alerting path — KS alerting uses the classic
asymptotic critical-value formula directly). Reports per-feature status and
an overall `ok`/`warn`/`alert`, embedded compactly in every
`/stream/{id}/push` response and available in full detail at
`GET /drift/{entity_id}`.

## The causality guarantee

At every layer, a score for timestep `t` may depend only on `x[<= t]` (plus
statistics fit once, at training time, on train data only):

- **Windowing**: a window's score is assigned to its last timestep, and
  `window_scores_to_points` only pads the *unreachable* first `window - 1`
  points with the first available (still causal) score — it never looks
  forward.
- **Baselines and deep models**: tested directly — `test_causality_*` in
  `tests/test_baselines.py`, `test_lstm_ae.py`, and `test_transformer_ae.py`
  perturbs the back half of a test series by `+100` and asserts every score
  *before* the perturbed region is byte-identical to the unperturbed run.
- **Streaming**: by construction (the ring buffer only ever holds points
  already pushed) and additionally proven equal to the (already-causal)
  batch path.
- **Normalization**: every detector's scaler (`mean_`/`std_`) is fit on
  `train` only, never on test or train+test combined.
