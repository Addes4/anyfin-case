"""Offline corpus quality, safety and metamorphic checks: python3 test_synthetic.py."""
import random
import unittest
from unittest.mock import patch

import classify as C
import synthetic as S


class SyntheticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = S.generate()
        cls.rows = [r["row"] for r in cls.records]

    def test_generation_is_reproducible(self):
        self.assertEqual(self.records, S.generate())
        self.assertNotEqual(self.records, S.generate(seed=1))

    def test_schema_and_category_coverage(self):
        self.assertEqual(len({r["id"] for r in self.rows}), len(self.rows))
        self.assertEqual({r["expected"] for r in self.records}, C.VALID)
        fields = set(C.read(C.TRAIN)[0]) - {"category"}
        self.assertTrue(all(set(r) == fields for r in self.rows))
        official = {r["id"] for r in C.read(C.TRAIN) + C.read("data/validation.csv")}
        self.assertTrue(official.isdisjoint(r["id"] for r in self.rows))

    def test_rules_do_not_make_wrong_assertions(self):
        report = S.evaluate(self.records, S.rules_predictions(self.rows))
        self.assertEqual(report["overall"]["wrong"], 0, report["errors"][:10])
        self.assertGreater(report["overall"]["correct"], 200)

    def test_row_order_does_not_change_rules(self):
        shuffled = self.rows[:]
        random.Random(100).shuffle(shuffled)
        self.assertEqual(S.rules_predictions(self.rows), S.rules_predictions(shuffled))

    def test_customers_do_not_share_transfer_evidence(self):
        # Same amount/date, but different owners: cannot be a transfer pair.
        rows = [dict(self.rows[0], id="a", customer="owner1", account="one", text="Överföring 111111", amount_sek="-500"),
                dict(self.rows[0], id="b", customer="owner2", account="two", text="Överföring 222222", amount_sek="500")]
        self.assertEqual(C.pair_transfers(rows), {})

    def test_official_rule_decisions_remain_correct(self):
        train = C.read(C.TRAIN)
        predictions = S.rules_predictions(train)
        for r in train:
            p = predictions[r["id"]]
            self.assertIn(p, (r["category"], "unknown"), r)

    def test_unknown_earns_zero_in_both_scorers(self):
        records = [{"row": {"id": "x"}, "expected": "unknown"}]
        self.assertEqual(S.metrics(records, {"x": "unknown"})["score"], 0)
        self.assertEqual(C.score({"x": "unknown"}, [{"id": "x", "category": "unknown"}])["score"], 0)
        self.assertEqual(C.score({}, [])["score"], 0)

    def test_predict_receives_no_synthetic_answer_keys(self):
        # Exercise actual predict() orchestration with an abstaining model, offline.
        captured = []
        def abstain(items, examples):
            captured.extend(items)
            return {C.key_of(it): "unknown" for it in items}
        with patch.object(C, "llm_label", abstain):
            predictions = C.predict(self.rows, C.read(C.TRAIN))
        self.assertEqual(len(predictions), len(self.rows))
        self.assertTrue(captured)
        self.assertTrue(all(set(it) == {"text", "dir", "acct", "size", "monthly"} for it in captured))
        self.assertTrue(all("ANNA" not in it["text"] for it in captured))


if __name__ == "__main__":
    unittest.main()
