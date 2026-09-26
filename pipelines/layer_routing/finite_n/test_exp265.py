"""Operator and endpoint regression tests without model loading."""
import unittest
import numpy as np
import pandas as pd
from common import arms
from evaluate import metrics, select, bootstrap
from score import DIMENSIONS, parse

class AnalysisTests(unittest.TestCase):
    def test_arm_ids_unique(self):
        self.assertEqual(len(arms()), 18)
        self.assertEqual(len(set(arms())), 18)

    def test_failure_and_unjudgeable_are_retained(self):
        j = {k: "PASS" for k in DIMENSIONS}
        j.update(target="YES", subject_role="YES")
        row = {"raw_fc": "B", "raw_guard": "YES", "pair": {"judgment": j}}
        self.assertEqual(metrics(row)["Q"], 1)
        j["objects"] = "UNJUDGEABLE"
        self.assertEqual(metrics(row)["Q"], 0)
        self.assertEqual(metrics(row)["preservation_unjudgeable"], 1)
        row["raw_fc"] = "N"
        self.assertEqual(metrics(row)["Y"], 0)

    def test_calibration_tie_rule(self):
        rows = [{"arm": a, "Y":1., "H":1., "J":1., "P":1., "Q":1., "T":1., "Q_visible":1.,
                 "actual_norm": 0.1 if a.endswith("d0") else 1.} for a in arms()]
        self.assertEqual(select(pd.DataFrame(rows)), {"F":"F_d0", "M":"M_d0", "A":"A_d0"})

    def test_bootstrap_pairs(self):
        self.assertEqual(bootstrap(np.zeros(12))["ci95"], [0.,0.])
        self.assertEqual(bootstrap(np.arange(12)), bootstrap(np.arange(12)))

    def test_parser_rejects_partial(self):
        with self.assertRaises(ValueError):
            parse('{"target":"YES"}')

class OperatorTests(unittest.TestCase):
    def test_shapes_support_budget_and_zero(self):
        import torch
        from model import build_conditionings, match_budget
        torch.manual_seed(265)
        neutral = (torch.randn(1,8,32)*4).to(torch.bfloat16)
        stereo = torch.randn(1,7,32).to(torch.bfloat16)
        anti = torch.randn(1,9,32).to(torch.bfloat16)
        unit = torch.randn(32)
        unit /= unit.norm()
        edits, meta = build_conditionings(neutral, stereo, anti, unit, 10., [2,3])
        self.assertEqual(set(edits),set(arms()))
        for arm,info in meta.items():
            if info["requested_norm"] is not None:
                self.assertLessEqual(abs(info["actual_norm"]/info["requested_norm"]-1), .05)
            if arm.startswith(("A","M")):
                self.assertTrue(set(info["active_rows"]).issubset({2,3}))
        zero,_ = match_budget(neutral,torch.ones_like(neutral),0)
        self.assertTrue(torch.equal(zero,neutral))

if __name__ == "__main__":
    unittest.main()
