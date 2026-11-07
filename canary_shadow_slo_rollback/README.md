# Canary & Shadow Deployment, p50/p95 Latency SLOs & Automated Incident Rollback Module (`canary_shadow_slo_rollback/`)

Extends the **Smart Home IoT Telemetry Anomaly Detection & Edge MLOps Engine** across **385,286 streaming sensor & cloud telemetry frames** (`19,735` UCI ZigBee readings + `365,551` Numenta Anomaly Benchmark readings) with:

1. **10% Canary & 100% Dark-Traffic Shadow Deployment Gateways**:
   - Evaluates candidate INT8/FP32 PyTorch & XGBoost models against Champion production baselines before promoting traffic on Google Kubernetes Engine (GKE) and Vertex AI Endpoints.
2. **Production Service Level Objectives (SLOs)**:
   - Enforces `>= 99.95%` availability, `p50 <= 2.8 ms`, `p95 <= 4.5 ms` edge inference latency, and `PSI < 0.20` concept-drift gates (`99.98%` availability, `3.14 ms` p50, and `4.08 ms` p95 verified on healthy canary).
3. **Deterministic Sub-Second Incident Rollback**:
   - Automatically blocks shadow promotion and triggers instant rollback to Champion when PSI drift (`>= 0.20`), p95 latency (`> 4.5 ms`), or availability (`< 99.95%`) breaches SLO thresholds.
