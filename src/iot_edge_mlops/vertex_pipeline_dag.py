"""GCP Vertex AI Pipelines (Kubeflow / TFX) DAG & Model Registry Governance Manifest."""

from __future__ import annotations

from typing import Dict, List


def build_vertex_kubeflow_pipeline_manifest(
    project_id: str = "cloud-mlops-telemetry-prod",
    region: str = "us-central1",
    bucket_uri: str = "gs://iot-edge-mlops-artifacts",
) -> Dict[str, object]:
    """Generate a reproducible Kubeflow / TFX Vertex AI Pipeline specification and promotion gates."""
    stages: List[Dict[str, object]] = [
        {
            "step_id": "01_bigquery_dataflow_ingestion",
            "component": "BigQueryToFeatureStoreIngestOp",
            "inputs": {"bq_table": f"{project_id}.iot_telemetry_warehouse.sensor_engineered_features"},
            "outputs": {"featurestore_entity": "iot_smart_sensor_featurestore/device_sensor_stream"},
        },
        {
            "step_id": "02_tfx_example_validator_and_schema_gate",
            "component": "TFXExampleValidatorOp",
            "contracts": ["non_null_check", "timestamp_monotonicity", "psi_drift_below_0.20"],
        },
        {
            "step_id": "03_gpu_model_training_and_tuning",
            "component": "VertexAICustomTrainingJobOp",
            "accelerator_type": "NVIDIA_TESLA_T4",
            "accelerator_count": 1,
            "frameworks": ["PyTorch", "XGBoost", "Scikit-Learn"],
        },
        {
            "step_id": "04_int8_edge_quantization_and_parity_gate",
            "component": "EdgeQuantizationExportOp",
            "targets": ["ONNX_INT8", "Edge_Runtime"],
            "gates": {"min_roc_auc": 0.94, "max_p99_latency_ms": 5.0, "min_compression_ratio": 3.5},
        },
        {
            "step_id": "05_vertex_model_registry_and_monitoring",
            "component": "VertexModelRegistryAndMonitoringDeployOp",
            "artifact_uri": f"{bucket_uri}/models/iot_anomaly_detector_int8",
            "drift_monitoring": {"psi_threshold": 0.20, "sampling_rate": 1.0, "alert_channel": "pubsub://mlops-drift-alerts"},
        },
    ]
    return {
        "pipeline_name": "iot-edge-telemetry-vertex-mlops-pipeline",
        "orchestrator": "Vertex AI Pipelines (Kubeflow v2 / TFX)",
        "project_id": project_id,
        "region": region,
        "artifact_bucket": bucket_uri,
        "stages": stages,
    }
