#!/usr/bin/env python3
"""Unit and Regression Tests for Canary/Shadow Deployment & SLO Rollback Governance."""

from __future__ import annotations

import unittest

from canary_shadow_evaluator import evaluate_canary_and_shadow_deployment


class TestCanaryShadowSLOGovernance(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = evaluate_canary_and_shadow_deployment()

    def test_01_corpus_scale_accounting(self) -> None:
        self.assertEqual(
            self.report["uci_zigbee_frames"] + self.report["nab_streaming_frames"],
            self.report["telemetry_corpus_frames"],
        )
        self.assertEqual(self.report["telemetry_corpus_frames"], 385286)

    def test_02_champion_slo_compliance(self) -> None:
        champ = self.report["champion_metrics"]
        slo = self.report["slo_targets"]
        self.assertLessEqual(champ["p95_latency_ms"], slo["p95_latency_ms_max"])
        self.assertGreaterEqual(champ["availability_pct"], slo["availability_pct_min"])
        self.assertGreaterEqual(champ["roc_auc"], 0.96)

    def test_03_shadow_traffic_mirror_parity(self) -> None:
        sh = self.report["shadow_mirror_evaluation"]
        self.assertEqual(sh["traffic_mirror_pct"], 100.0)
        self.assertGreaterEqual(sh["decision_parity_pct"], 98.5)
        self.assertLess(sh["mean_probability_divergence"], 0.02)
        self.assertTrue(sh["promotion_gate_passed"])

    def test_04_degraded_canary_slo_breach_and_auto_rollback(self) -> None:
        can = self.report["canary_incident_rollback"]
        self.assertTrue(can["slo_breach_detected"])
        self.assertEqual(can["incident_action"], "AUTO_ROLLBACK_TO_CHAMPION_V1_4_0")
        self.assertLess(can["rollback_drain_latency_ms"], 500.0)

    def test_05_lineage_telemetry_and_react_contract(self) -> None:
        lin = self.report["model_lineage_telemetry"]
        self.assertEqual(len(lin["lineage_artifact_sha256_prefix"]), 16)
        self.assertEqual(lin["react_dashboard_schema_version"], "2.1.0")


if __name__ == "__main__":
    unittest.main(verbosity=2)
