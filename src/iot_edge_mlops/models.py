"""Deep Neural MLP, Gradient Boosted Ensemble, and Dynamic INT8 Edge Quantization Engine."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Dict
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neural_network import MLPClassifier


@dataclass(frozen=True)
class EdgeQuantizationMetrics:
    """Benchmark metrics comparing FP64/FP32 baseline model weights to INT8 edge-quantized inference."""

    fp32_roc_auc: float
    int8_roc_auc: float
    int8_f1_score: float
    int8_precision: float
    int8_recall: float
    compression_ratio: float
    p99_edge_latency_ms: float
    max_logit_parity_delta: float


class EdgeQuantizedNeuralDetector:
    """Two-layer neural anomaly detector with symmetric per-tensor INT8 weight quantization for edge devices."""

    def __init__(self, hidden_layer_sizes: tuple[int, int] = (64, 32), random_state: int = 42) -> None:
        self.mlp = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=1e-3,
            max_iter=120,
            random_state=random_state,
        )
        self.gbt = HistGradientBoostingClassifier(
            max_depth=5,
            learning_rate=0.08,
            max_iter=120,
            random_state=random_state,
        )
        self.quantized_weights_: list[tuple[np.ndarray, float]] = []
        self.biases_: list[np.ndarray] = []

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "EdgeQuantizedNeuralDetector":
        self.mlp.fit(X_train, y_train)
        self.gbt.fit(X_train, y_train)
        self._quantize_weights_int8()
        return self

    def _quantize_weights_int8(self) -> None:
        """Apply symmetric per-layer INT8 quantization: W_q = clip(round(W / scale), -127, 127)."""
        self.quantized_weights_ = []
        self.biases_ = []
        for W, b in zip(self.mlp.coefs_, self.mlp.intercepts_):
            max_abs = float(np.max(np.abs(W)))
            scale = max_abs / 127.0 if max_abs > 1e-12 else 1.0
            W_int8 = np.clip(np.round(W / scale), -127, 127).astype(np.int8)
            self.quantized_weights_.append((W_int8, scale))
            self.biases_.append(b.astype(np.float32))

    def predict_proba_fp32(self, X: np.ndarray) -> np.ndarray:
        mlp_prob = self.mlp.predict_proba(X)[:, 1]
        gbt_prob = self.gbt.predict_proba(X)[:, 1]
        return 0.55 * mlp_prob + 0.45 * gbt_prob

    def predict_proba_int8_edge(self, X: np.ndarray) -> np.ndarray:
        """Execute INT8-dequantized neural forward pass blended with tree ensemble for low-latency edge scoring."""
        h = X.astype(np.float32)
        for idx, ((W_int8, scale), b) in enumerate(zip(self.quantized_weights_, self.biases_)):
            W_deq = W_int8.astype(np.float32) * np.float32(scale)
            h = h @ W_deq + b
            if idx < len(self.quantized_weights_) - 1:
                h = np.maximum(0.0, h)
        logits = h.ravel()
        mlp_int8_prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        gbt_prob = self.gbt.predict_proba(X)[:, 1]
        return 0.55 * mlp_int8_prob + 0.45 * gbt_prob

    def evaluate_and_benchmark(self, X_test: np.ndarray, y_test: np.ndarray) -> EdgeQuantizationMetrics:
        prob_fp32 = self.predict_proba_fp32(X_test)
        prob_int8 = self.predict_proba_int8_edge(X_test)
        preds_int8 = (prob_int8 >= 0.45).astype(int)

        # Measure p99 single-batch (64 telemetry events) edge inference latency
        latencies_ms = []
        batch = X_test[: min(64, len(X_test))]
        for _ in range(80):
            t0 = time.perf_counter()
            _ = self.predict_proba_int8_edge(batch)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        fp32_bytes = sum(w.nbytes for w in self.mlp.coefs_)
        int8_bytes = sum(w_q.nbytes + 4 for w_q, _ in self.quantized_weights_)
        compression_ratio = round(fp32_bytes / max(int8_bytes, 1), 2)

        return EdgeQuantizationMetrics(
            fp32_roc_auc=round(float(roc_auc_score(y_test, prob_fp32)), 4),
            int8_roc_auc=round(float(roc_auc_score(y_test, prob_int8)), 4),
            int8_f1_score=round(float(f1_score(y_test, preds_int8)), 4),
            int8_precision=round(float(precision_score(y_test, preds_int8)), 4),
            int8_recall=round(float(recall_score(y_test, preds_int8)), 4),
            compression_ratio=compression_ratio,
            p99_edge_latency_ms=round(float(np.percentile(latencies_ms, 99)), 3),
            max_logit_parity_delta=round(float(np.max(np.abs(prob_fp32 - prob_int8))), 5),
        )
