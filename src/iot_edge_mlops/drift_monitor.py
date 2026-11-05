"""Vertex AI Model Monitoring: Population Stability Index (PSI) & Jensen-Shannon Drift Detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass(frozen=True)
class DriftReport:
    """Structured model monitoring report for Vertex AI Model Monitoring and GCS logging."""

    feature_psi: Dict[str, float]
    mean_psi: float
    max_psi: float
    jensen_shannon_divergence: float
    retraining_triggered: bool
    psi_threshold: float


def compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Compute Population Stability Index: PSI = sum((p_i - q_i) * ln(p_i / q_i))."""
    quantiles = np.linspace(0.0, 100.0, bins + 1)
    edges = np.percentile(reference, quantiles)
    edges[0] -= 1e-6
    edges[-1] += 1e-6
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    eps = 1e-4
    ref_dist = np.maximum(ref_counts / max(len(reference), 1), eps)
    cur_dist = np.maximum(cur_counts / max(len(current), 1), eps)
    psi_value = float(np.sum((cur_dist - ref_dist) * np.log(cur_dist / ref_dist)))
    return round(max(0.0, psi_value), 4)


def compute_jensen_shannon(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Compute symmetric Jensen-Shannon divergence between reference and live feature distributions."""
    edges = np.linspace(
        min(float(np.min(reference)), float(np.min(current))) - 1e-6,
        max(float(np.max(reference)), float(np.max(current))) + 1e-6,
        bins + 1,
    )
    p, _ = np.histogram(reference, bins=edges)
    q, _ = np.histogram(current, bins=edges)
    p = np.maximum(p / max(np.sum(p), 1), 1e-6)
    q = np.maximum(q / max(np.sum(q), 1), 1e-6)
    p /= np.sum(p)
    q /= np.sum(q)
    m = 0.5 * (p + q)
    js = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))
    return round(float(max(0.0, js)), 5)


def evaluate_telemetry_drift(
    X_ref: np.ndarray,
    X_cur: np.ndarray,
    feature_names: list[str],
    psi_threshold: float = 0.20,
) -> DriftReport:
    """Evaluate feature-level PSI and Jensen-Shannon drift across reference and production windows."""
    psi_map: Dict[str, float] = {}
    js_values: list[float] = []
    for idx, name in enumerate(feature_names):
        psi_val = compute_psi(X_ref[:, idx], X_cur[:, idx])
        psi_map[name] = psi_val
        js_values.append(compute_jensen_shannon(X_ref[:, idx], X_cur[:, idx]))

    mean_psi = round(float(np.mean(list(psi_map.values()))), 4)
    max_psi = round(float(np.max(list(psi_map.values()))), 4)
    mean_js = round(float(np.mean(js_values)), 5)
    return DriftReport(
        feature_psi=psi_map,
        mean_psi=mean_psi,
        max_psi=max_psi,
        jensen_shannon_divergence=mean_js,
        retraining_triggered=bool(max_psi >= psi_threshold),
        psi_threshold=psi_threshold,
    )
