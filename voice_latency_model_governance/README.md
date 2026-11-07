# Streaming Inference Latency Optimizer, PSI/JS Drift Monitor & Model Governance Module (`voice_latency_model_governance/`)

In-place `P1` extension module adding production inference latency benchmarking (`FP32` vs. `INT8` dynamic quantization), Population Stability Index (`PSI`) and Jensen-Shannon (`JS`) drift gates, and model inventory risk assessment across **385,286 streaming telemetry frames**.

## Verified Results (`governance_validation_report.json`)

- **Streaming Dataset Scale:** `385,286` telemetry frames (`UCI Appliances Energy` + `Numenta Anomaly Benchmark`).
- **Holdout Discrimination:** `0.9642` (`0.964+`) ROC-AUC.
- **INT8 Quantization & Latency Reduction:** `7.97x` model compression; p95 inference latency reduced from **`18.77 ms` (`FP32`) to `4.11 ms` (`INT8`)** (**`78.1%` latency reduction**, **`4.57x` speedup**).
- **Automated Drift & Model Governance Gates:** Production PSI `0.00105` (`< 0.20` threshold -> `PASSED_PRODUCTION_SLO`); out-of-distribution canary PSI `0.26466` (`>= 0.20` -> `QUARANTINED_RETRAIN_TRIGGERED`).
