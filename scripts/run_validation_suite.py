"""End-to-End Validation & Benchmark Runner for IoT Edge Telemetry & Vertex AI MLOps Engine."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iot_edge_mlops import (  # noqa: E402
    IoTTelemetryFeatureEngineer,
    EdgeQuantizedNeuralDetector,
    evaluate_telemetry_drift,
    run_glrt_mle_telemetry_benchmark,
    build_vertex_kubeflow_pipeline_manifest,
)


def main() -> None:
    df = IoTTelemetryFeatureEngineer.generate_realistic_iot_telemetry(n_samples=6000, n_devices=25, random_state=42)
    split_idx = int(len(df) * 0.75)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    fe = IoTTelemetryFeatureEngineer(rolling_window=6)
    X_train, y_train = fe.fit_transform(train_df)
    X_test, y_test = fe.transform(test_df)

    detector = EdgeQuantizedNeuralDetector(hidden_layer_sizes=(64, 32), random_state=42)
    detector.fit(X_train, y_train)
    metrics = detector.evaluate_and_benchmark(X_test, y_test)

    # Evaluate healthy vs drifted production windows
    healthy_drift = evaluate_telemetry_drift(X_train, X_test, fe.feature_names_, psi_threshold=0.20)
    X_drifted = X_test.copy()
    X_drifted[:, 0] += 1.65  # Simulate thermal sensor calibration drift
    incident_drift = evaluate_telemetry_drift(X_train, X_drifted, fe.feature_names_, psi_threshold=0.20)

    glrt_summary = run_glrt_mle_telemetry_benchmark(random_state=42)
    dag_manifest = build_vertex_kubeflow_pipeline_manifest()

    report = {
        "dataset_summary": {
            "total_evaluated_rows": int(len(df)),
            "engineered_feature_count": len(fe.feature_names_),
            "anomaly_rate": round(float(df["is_anomaly"].mean()), 4),
        },
        "model_and_edge_quantization_metrics": asdict(metrics),
        "vertex_model_monitoring_drift": {
            "healthy_window_mean_psi": healthy_drift.mean_psi,
            "healthy_window_max_psi": healthy_drift.max_psi,
            "healthy_retraining_triggered": healthy_drift.retraining_triggered,
            "drifted_window_max_psi": incident_drift.max_psi,
            "drifted_retraining_triggered": incident_drift.retraining_triggered,
        },
        "multi_sensor_glrt_signal_detection": asdict(glrt_summary),
        "vertex_pipeline_stage_count": len(dag_manifest["stages"]),
        "feature_store_spec": fe.export_vertex_feature_store_spec(),
    }

    artifacts_dir = ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifacts_dir / "validation_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
