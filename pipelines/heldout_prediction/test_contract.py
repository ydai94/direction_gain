import unittest
from common import CODE,read_json
from run import selected
class Contract(unittest.TestCase):
    def test_cohort_partition(self):
        fits=read_json(CODE/'predictors_frozen.json')
        for m in ['sd3','flux','qwen']:
            n=83 if m=='qwen' else 42;rows=[r for i in range(n) for r in selected(m,i)]
            self.assertEqual(len(rows),1314);keys={r['triplet_key'] for r in rows};self.assertEqual(len(keys),1314)
            self.assertFalse(keys&set(fits['models'][m]['triplet_keys']))
    def test_null_frozen(self):
        for n in read_json(CODE/'development_calibration.json').values():self.assertGreater(n['sd'],0)
    def test_score_decoder(self):
        from score_engine import parse
        self.assertIs(parse('false','alignment'),False)
        self.assertEqual(parse('0','bias'),0)
        with self.assertRaises(ValueError):parse('0 because','bias')
