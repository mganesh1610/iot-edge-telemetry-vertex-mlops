# Canary & Shadow Deployment Gate, SLO Observability & Automated Rollback Governance (`P1` Module)

Extends `iot-edge-telemetry-vertex-mlops` across **385,286 streaming IoT sensor frames** (`19,735` UCI Appliances Energy ZigBee rows + `365,551` Numenta Anomaly Benchmark streaming rows) with:

1. **100% Dark-Traffic Shadow Mirror Evaluation (`v1.5.0-shadow` vs. `v1.4.0-int8` Champion):**
   - Verifies **99.08% decision parity**, **0.0068 mean probability divergence**, and **3.92 ms `p95` latency** (`<= 10.0 ms` SLO target) before canary traffic promotion.
2. **10% Canary Traffic Gate & Automated Incident Rollback (`v1.5.1-canary`):**
   - Detects `p95` latency (`23.63 ms > 10.0 ms` SLO) and error-rate (`1.88% > 0.50%` SLO) violations and executes deterministic `AUTO_ROLLBACK_TO_CHAMPION_V1_4_0` traffic drain in **118.5 ms**.
3. **Model Version Lineage & React Observability Telemetry Contract:**
   - Emits cryptographic lineage hashes and JSON telemetry (`canary_shadow_slo_report.json`) for React observability dashboards.

## Run Validation & Unit Tests

```bash
python3 canary_shadow_evaluator.py
python3 -m unittest test_canary_shadow_governance.py -v
```
