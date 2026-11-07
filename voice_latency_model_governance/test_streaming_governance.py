#!/usr/bin/env python3
"""Unit tests for Streaming Inference Latency Optimizer & Model Governance Module."""

import unittest
from pathlib import Path
import sys

MOD_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MOD_DIR))

from streaming_inference_governance import run_streaming_governance_benchmark


class TestStreamingGovernance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = run_streaming_governance_benchmark()

    def test_01_int8_latency_slo(self):
        self.assertLess(self.report["int8_p95_latency_ms"], 4.2)
        self.assertGreater(self.report["p95_latency_reduction_pct"], 75.0)
        self.assertEqual(self.report["model_compression_ratio_x"], 7.97)

    def test_02_psi_and_js_drift_gates(self):
        self.assertLess(self.report["stable_psi"], 0.05)
        self.assertGreaterEqual(self.report["canary_drift_psi"], 0.20)
        self.assertTrue(self.report["canary_retrain_triggered"])

    def test_03_model_inventory_governance_records(self):
        inv = self.report["model_inventory_registry"]
        self.assertEqual(len(inv), 2)
        self.assertEqual(inv[0]["deployment_gate"], "PASSED_PRODUCTION_SLO")
        self.assertEqual(inv[1]["deployment_gate"], "QUARANTINED_RETRAIN_TRIGGERED")


if __name__ == "__main__":
    unittest.main()
