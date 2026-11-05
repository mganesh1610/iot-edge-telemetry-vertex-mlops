"""Low-Latency Telemetry Microservice & Duplicate-Safe Cloud Ingestion Gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Any
import numpy as np
import pandas as pd

from .drift_monitor import evaluate_telemetry_drift


@dataclass(frozen=True)
class TelemetryMicroserviceHealthReport:
  """Operational telemetry ingestion and drift-gate health report."""

  stream_batch_rows: int
  deduplicated_rows_inserted: int
  duplicate_rows_filtered: int
  stale_streams_flagged: int
  critical_alert_streams: int
  max_psi: float
  mean_psi: float
  psi_contract_passed: bool
  p95_ingestion_latency_ms: float

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)


def deduplicate_staging_telemetry(
    staging_df: pd.DataFrame,
    existing_ids: set[str],
    id_col: str = "reading_id",
) -> tuple[pd.DataFrame, int]:
  """Filter overlapping rolling-window telemetry readings before warehouse merge."""
  if id_col not in staging_df.columns:
    raise ValueError(f"Missing required deduplication key column: '{id_col}'")
  dedup_stage = staging_df.drop_duplicates(subset=[id_col], keep="first")
  mask_new = ~dedup_stage[id_col].astype(str).isin(existing_ids)
  filtered_df = dedup_stage.loc[mask_new].copy()
  duplicates_removed = int(len(staging_df) - len(filtered_df))
  return filtered_df, duplicates_removed


def evaluate_telemetry_microservice_batch(
    reference_features: np.ndarray,
    current_features: np.ndarray,
    feature_names: list[str],
    staging_df: pd.DataFrame,
    existing_ids: set[str],
    psi_threshold: float = 0.20,
    stale_minutes_threshold: float = 60.0,
    n_timing_repeats: int = 12,
) -> TelemetryMicroserviceHealthReport:
  """Run duplicate-safe ingestion, stale-stream triage, and PSI drift contract checks with empirical latency timing."""
  timings_ms: list[float] = []
  clean_df = staging_df
  dup_count = 0
  drift_report = None

  for _ in range(max(n_timing_repeats, 1)):
    t0 = time.perf_counter()
    clean_df, dup_count = deduplicate_staging_telemetry(
        staging_df, existing_ids
    )
    drift_report = evaluate_telemetry_drift(
        reference_features,
        current_features,
        feature_names,
        psi_threshold=psi_threshold,
    )
    stale_count = int(
        (clean_df["minutes_since_last_ping"] > stale_minutes_threshold).sum()
    )
    critical_count = int(
        (
            (clean_df["sensor_reading"] < clean_df["low_limit"])
            | (clean_df["sensor_reading"] > clean_df["high_limit"])
        ).sum()
    )
    timings_ms.append((time.perf_counter() - t0) * 1000.0)

  assert drift_report is not None
  p95_ms = round(float(np.percentile(timings_ms, 95)), 3)
  return TelemetryMicroserviceHealthReport(
      stream_batch_rows=int(len(staging_df)),
      deduplicated_rows_inserted=int(len(clean_df)),
      duplicate_rows_filtered=dup_count,
      stale_streams_flagged=stale_count,
      critical_alert_streams=critical_count,
      max_psi=drift_report.max_psi,
      mean_psi=drift_report.mean_psi,
      psi_contract_passed=not drift_report.retraining_triggered,
      p95_ingestion_latency_ms=p95_ms,
  )
