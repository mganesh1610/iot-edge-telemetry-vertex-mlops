"""Smart Home & Industrial IoT Telemetry Feature Engineering & Vertex AI Feature Store Pipeline."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureStoreEntityConfig:
    """Configuration for GCP Vertex AI Feature Store and BigQuery offline/online tables."""

    project_id: str = "cloud-mlops-telemetry-prod"
    region: str = "us-central1"
    featurestore_id: str = "iot_smart_sensor_featurestore"
    entity_type_id: str = "device_sensor_stream"
    bq_dataset: str = "iot_telemetry_warehouse"
    bq_table: str = "sensor_engineered_features"
    online_serving_ttl_days: int = 14


class IoTTelemetryFeatureEngineer:
    """Object-oriented feature engineering pipeline for high-dimensional IoT sensor telemetry."""

    REQUIRED_COLUMNS: Tuple[str, ...] = (
        "device_id",
        "event_timestamp",
        "temp_indoor_c",
        "humidity_indoor_pct",
        "power_draw_watts",
        "vibration_rms_g",
        "pressure_hpa",
        "motion_pir_score",
        "acoustic_db",
        "is_anomaly",
    )

    def __init__(self, rolling_window: int = 6) -> None:
        if rolling_window < 2:
            raise ValueError("rolling_window must be >= 2")
        self.rolling_window = rolling_window
        self.feature_names_: List[str] = []
        self.train_mean_: np.ndarray | None = None
        self.train_std_: np.ndarray | None = None

    @staticmethod
    def generate_realistic_iot_telemetry(
        n_samples: int = 6000, n_devices: int = 25, random_state: int = 42
    ) -> pd.DataFrame:
        """Generate deterministic smart-home & edge sensor telemetry calibrated to UCI Appliances Energy & NAB distributions."""
        rng = np.random.default_rng(random_state)
        device_ids = [f"DEV-{1000 + (i % n_devices)}" for i in range(n_samples)]
        base_time = pd.Timestamp("2026-01-01T00:00:00Z")
        timestamps = [base_time + pd.Timedelta(minutes=10 * i) for i in range(n_samples)]

        phase = np.linspace(0, 24 * np.pi, n_samples)
        temp_indoor_c = 21.2 + 2.1 * np.sin(phase) + rng.normal(0.0, 0.45, n_samples)
        humidity_indoor_pct = 42.5 - 3.8 * np.sin(phase) + rng.normal(0.0, 1.8, n_samples)
        pressure_hpa = 1013.2 + 3.5 * np.cos(phase * 0.25) + rng.normal(0.0, 0.9, n_samples)

        # Inject structured multi-sensor anomalies (8.5% positive rate)
        labels = (rng.uniform(0.0, 1.0, n_samples) < 0.085).astype(int)

        power_draw_watts = (
            95.0
            + 28.0 * np.maximum(0.0, np.sin(phase))
            + rng.normal(0.0, 9.5, n_samples)
            + labels * rng.normal(145.0, 26.0, n_samples)
        )
        vibration_rms_g = (
            0.12
            + 0.03 * np.abs(np.cos(phase))
            + rng.normal(0.0, 0.018, n_samples)
            + labels * rng.normal(0.24, 0.05, n_samples)
        )
        motion_pir_score = np.clip(
            0.22 + 0.18 * np.sin(phase) + rng.normal(0.0, 0.08, n_samples) + labels * rng.normal(0.42, 0.11, n_samples),
            0.0,
            1.0,
        )
        acoustic_db = (
            38.0
            + 4.5 * np.abs(np.sin(phase))
            + rng.normal(0.0, 2.2, n_samples)
            + labels * rng.normal(16.5, 3.8, n_samples)
        )
        temp_indoor_c = temp_indoor_c + labels * rng.normal(3.6, 0.9, n_samples)

        return pd.DataFrame(
            {
                "device_id": device_ids,
                "event_timestamp": timestamps,
                "temp_indoor_c": np.round(temp_indoor_c, 4),
                "humidity_indoor_pct": np.round(humidity_indoor_pct, 4),
                "power_draw_watts": np.round(power_draw_watts, 4),
                "vibration_rms_g": np.round(vibration_rms_g, 4),
                "pressure_hpa": np.round(pressure_hpa, 4),
                "motion_pir_score": np.round(motion_pir_score, 4),
                "acoustic_db": np.round(acoustic_db, 4),
                "is_anomaly": labels,
            }
        )

    def validate_schema(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required telemetry columns: {missing}")
        if df[list(self.REQUIRED_COLUMNS)].isnull().any().any():
            raise ValueError("Null values detected in required telemetry fields")

    def build_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Engineer rolling temporal statistics, z-deltas, and cross-sensor interaction features."""
        self.validate_schema(df)
        work = df.sort_values(["device_id", "event_timestamp"]).reset_index(drop=True).copy()

        raw_cols = [
            "temp_indoor_c",
            "humidity_indoor_pct",
            "power_draw_watts",
            "vibration_rms_g",
            "pressure_hpa",
            "motion_pir_score",
            "acoustic_db",
        ]

        for col in ["power_draw_watts", "vibration_rms_g", "acoustic_db", "temp_indoor_c"]:
            grp = work.groupby("device_id")[col]
            work[f"{col}_roll_mean"] = grp.transform(
                lambda s: s.rolling(self.rolling_window, min_periods=1).mean()
            )
            work[f"{col}_roll_std"] = grp.transform(
                lambda s: s.rolling(self.rolling_window, min_periods=1).std().fillna(0.0)
            )
            work[f"{col}_delta"] = grp.diff().fillna(0.0)

        # Interaction terms: electro-mechanical stress & thermal-humidity index
        work["electro_vib_interaction"] = work["power_draw_watts"] * work["vibration_rms_g"]
        work["acoustic_motion_product"] = work["acoustic_db"] * work["motion_pir_score"]
        work["thermal_humidity_ratio"] = work["temp_indoor_c"] / np.maximum(work["humidity_indoor_pct"], 1.0)

        feature_cols = [c for c in work.columns if c not in ("device_id", "event_timestamp", "is_anomaly")]
        self.feature_names_ = feature_cols
        X = work[feature_cols].to_numpy(dtype=np.float64)
        y = work["is_anomaly"].to_numpy(dtype=np.int64)
        return X, y, feature_cols

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X, y, _ = self.build_features(df)
        self.train_mean_ = X.mean(axis=0)
        self.train_std_ = np.where(X.std(axis=0) < 1e-8, 1.0, X.std(axis=0))
        X_scaled = (X - self.train_mean_) / self.train_std_
        return X_scaled, y

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if self.train_mean_ is None or self.train_std_ is None:
            raise RuntimeError("IoTTelemetryFeatureEngineer must be fitted before calling transform().")
        X, y, _ = self.build_features(df)
        X_scaled = (X - self.train_mean_) / self.train_std_
        return X_scaled, y

    @staticmethod
    def export_vertex_feature_store_spec(config: FeatureStoreEntityConfig | None = None) -> Dict[str, object]:
        cfg = config or FeatureStoreEntityConfig()
        return {
            "featurestore_config": asdict(cfg),
            "bigquery_source_uri": f"bq://{cfg.project_id}.{cfg.bq_dataset}.{cfg.bq_table}",
            "entity_id_column": "device_id",
            "feature_time_column": "event_timestamp",
        }
