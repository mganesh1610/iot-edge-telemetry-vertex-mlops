"""Unit tests for GKE Kubernetes autoscaling manifest and Bitbucket CI/CD MLOps promotion gate."""

from __future__ import annotations

import pytest
from iot_edge_mlops import (
    build_gke_telemetry_manifest,
    evaluate_bitbucket_cicd_promotion_gate,
)


def test_bitbucket_cicd_promotion_gate_passes_healthy_candidate() -> None:
  report = evaluate_bitbucket_cicd_promotion_gate(
      candidate_model_version="v2.1.0-int8",
      int8_roc_auc=0.968,
      p99_edge_latency_ms=3.1,
      compression_ratio=3.99,
      healthy_max_psi=0.042,
  )
  assert report.promote_to_gke_production is True
  assert report.target_gke_min_replicas == 3
  assert report.target_gke_max_replicas == 12


def test_bitbucket_cicd_promotion_gate_blocks_drifted_or_slow_candidate() -> (
    None
):
  report = evaluate_bitbucket_cicd_promotion_gate(
      candidate_model_version="v2.1.1-regressed",
      int8_roc_auc=0.968,
      p99_edge_latency_ms=7.4,
      compression_ratio=3.99,
      healthy_max_psi=0.28,
  )
  assert report.edge_latency_gate_passed is False
  assert report.psi_drift_gate_passed is False
  assert report.promote_to_gke_production is False
  assert report.target_gke_min_replicas == 0


def test_build_gke_telemetry_manifest_structure() -> None:
  manifest = build_gke_telemetry_manifest(min_replicas=3, max_replicas=12)
  assert manifest["kind"] == "Deployment"
  assert manifest["hpa"]["minReplicas"] == 3
  assert manifest["hpa"]["maxReplicas"] == 12
  with pytest.raises(ValueError):
    build_gke_telemetry_manifest(min_replicas=0, max_replicas=5)
