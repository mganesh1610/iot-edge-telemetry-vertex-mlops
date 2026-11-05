"""GKE Kubernetes Microservice Autoscaling & Bitbucket CI/CD MLOps Promotion Gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class GKEPromotionGateReport:
  """Bitbucket CI/CD & Vertex AI Model Registry promotion gate evaluation."""

  candidate_model_version: str
  roc_auc_gate_passed: bool
  edge_latency_gate_passed: bool
  compression_gate_passed: bool
  psi_drift_gate_passed: bool
  promote_to_gke_production: bool
  target_gke_min_replicas: int
  target_gke_max_replicas: int

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)


def evaluate_bitbucket_cicd_promotion_gate(
    candidate_model_version: str,
    int8_roc_auc: float,
    p99_edge_latency_ms: float,
    compression_ratio: float,
    healthy_max_psi: float,
    min_roc_auc: float = 0.940,
    max_latency_ms: float = 5.0,
    min_compression_ratio: float = 3.5,
    max_allowed_psi: float = 0.20,
) -> GKEPromotionGateReport:
  """Evaluate whether a candidate Vertex AI model passes Bitbucket CI/CD gates for GKE deployment."""
  auc_ok = int8_roc_auc >= min_roc_auc
  lat_ok = p99_edge_latency_ms <= max_latency_ms
  comp_ok = compression_ratio >= min_compression_ratio
  psi_ok = healthy_max_psi < max_allowed_psi
  promote = bool(auc_ok and lat_ok and comp_ok and psi_ok)
  return GKEPromotionGateReport(
      candidate_model_version=candidate_model_version,
      roc_auc_gate_passed=auc_ok,
      edge_latency_gate_passed=lat_ok,
      compression_gate_passed=comp_ok,
      psi_drift_gate_passed=psi_ok,
      promote_to_gke_production=promote,
      target_gke_min_replicas=3 if promote else 0,
      target_gke_max_replicas=12 if promote else 0,
  )


def build_gke_telemetry_manifest(
    service_name: str = "iot-telemetry-inference-svc",
    image_tag: str = "us-central1-docker.pkg.dev/iot-mlops/edge-inference:v2.1.0",
    min_replicas: int = 3,
    max_replicas: int = 12,
    target_cpu_utilization: int = 65,
) -> dict[str, Any]:
  """Construct a validated Google Kubernetes Engine (GKE) Deployment and HorizontalPodAutoscaler spec."""
  if min_replicas < 1 or max_replicas < min_replicas:
    raise ValueError("Invalid GKE replica bounds.")
  return {
      "apiVersion": "apps/v1",
      "kind": "Deployment",
      "metadata": {"name": service_name, "namespace": "telemetry-mlops"},
      "spec": {
          "replicas": min_replicas,
          "selector": {"matchLabels": {"app": service_name}},
          "template": {
              "metadata": {"labels": {"app": service_name}},
              "spec": {
                  "containers": [{
                      "name": "fastapi-int8-scorer",
                      "image": image_tag,
                      "ports": [{"containerPort": 8080}],
                      "resources": {
                          "requests": {"cpu": "500m", "memory": "512Mi"},
                          "limits": {"cpu": "1000m", "memory": "1Gi"},
                      },
                      "readinessProbe": {
                          "httpGet": {"path": "/healthz", "port": 8080},
                          "initialDelaySeconds": 5,
                          "periodSeconds": 10,
                      },
                  }]
              },
          },
      },
      "hpa": {
          "apiVersion": "autoscaling/v2",
          "kind": "HorizontalPodAutoscaler",
          "minReplicas": min_replicas,
          "maxReplicas": max_replicas,
          "targetCPUUtilizationPercentage": target_cpu_utilization,
      },
  }
