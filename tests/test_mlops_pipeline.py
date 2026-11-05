"""Unit and Integration Tests for IoT Edge Telemetry & Vertex AI MLOps Engine."""

from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iot_edge_mlops import (  # noqa: E402
    FeatureStoreEntityConfig,
    IoTTelemetryFeatureEngineer,
    EdgeQuantizedNeuralDetector,
    compute_psi,
    compute_jensen_shannon,
    evaluate_telemetry_drift,
    run_glrt_mle_telemetry_benchmark,
    build_vertex_kubeflow_pipeline_manifest,
)


def test_feature_engineering_output_shape_and_scaling() -> None:
    df = IoTTelemetryFeatureEngineer.generate_realistic_iot_telemetry(n_samples=600, n_devices=6, random_state=7)
    fe = IoTTelemetryFeatureEngineer(rolling_window=4)
    X_train, y_train = fe.fit_transform(df.iloc[:400])
    X_test, y_test = fe.transform(df.iloc[400:])

    assert X_train.shape == (400, 22)
    assert X_test.shape == (200, 22)
    assert y_train.shape == (400,)
    assert y_test.shape == (200,)
    assert np.allclose(X_train.mean(axis=0), 0.0, atol=1e-6)


def test_schema_validation_rejects_missing_columns() -> None:
    df = IoTTelemetryFeatureEngineer.generate_realistic_iot_telemetry(n_samples=50, n_devices=2)
    bad_df = df.drop(columns=["power_draw_watts"])
    fe = IoTTelemetryFeatureEngineer()
    with pytest.raises(ValueError, match="Missing required telemetry columns"):
        fe.build_features(bad_df)


def test_int8_edge_quantization_roc_auc_and_latency() -> None:
    df = IoTTelemetryFeatureEngineer.generate_realistic_iot_telemetry(n_samples=2000, n_devices=10, random_state=42)
    fe = IoTTelemetryFeatureEngineer(rolling_window=6)
    X_train, y_train = fe.fit_transform(df.iloc[:1500])
    X_test, y_test = fe.transform(df.iloc[1500:])

    detector = EdgeQuantizedNeuralDetector(hidden_layer_sizes=(64, 32), random_state=42)
    detector.fit(X_train, y_train)
    metrics = detector.evaluate_and_benchmark(X_test, y_test)

    assert metrics.int8_roc_auc >= 0.94
    assert metrics.compression_ratio >= 3.5
    assert metrics.p99_edge_latency_ms < 15.0
    assert metrics.max_logit_parity_delta < 0.05


def test_psi_and_jensen_shannon_drift_trigger() -> None:
    df = IoTTelemetryFeatureEngineer.generate_realistic_iot_telemetry(n_samples=1200, n_devices=8, random_state=11)
    fe = IoTTelemetryFeatureEngineer(rolling_window=5)
    X_ref, _ = fe.fit_transform(df.iloc[:600])
    X_clean, _ = fe.transform(df.iloc[600:])

    clean_report = evaluate_telemetry_drift(X_ref, X_clean, fe.feature_names_, psi_threshold=0.20)
    assert clean_report.retraining_triggered is False
    assert clean_report.mean_psi < 0.10

    X_drifted = X_clean.copy()
    X_drifted[:, 2] += 2.2
    drift_report = evaluate_telemetry_drift(X_ref, X_drifted, fe.feature_names_, psi_threshold=0.20)
    assert drift_report.retraining_triggered is True
    assert drift_report.max_psi >= 0.20
    assert compute_jensen_shannon(X_ref[:, 2], X_drifted[:, 2]) > 0.05


def test_glrt_mle_telemetry_detection_benchmark() -> None:
    summary = run_glrt_mle_telemetry_benchmark(random_state=42)
    assert summary.mle_delay_error_samples == 0
    assert summary.glrt_peak_lift_pct > 20.0
    assert summary.jammer_separation_ratio > 5.0


def test_vertex_pipeline_and_feature_store_manifests() -> None:
    fe = IoTTelemetryFeatureEngineer()
    fs_spec = fe.export_vertex_feature_store_spec(FeatureStoreEntityConfig())
    assert "bq://cloud-mlops-telemetry-prod.iot_telemetry_warehouse.sensor_engineered_features" in str(
        fs_spec["bigquery_source_uri"]
    )
    dag = build_vertex_kubeflow_pipeline_manifest()
    assert len(dag["stages"]) == 5
    assert dag["orchestrator"] == "Vertex AI Pipelines (Kubeflow v2 / TFX)"
