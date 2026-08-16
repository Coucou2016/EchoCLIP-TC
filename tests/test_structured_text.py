import unittest

from echoclip.prompts import ZERO_SHOT_PROMPTS
from echoclip.structured_text import (
    captions_from_measurements,
    fill_numeric_template,
    measurements_to_text,
    pair_record,
)


class TestStructuredText(unittest.TestCase):
    def test_ef_uses_official_templates_only(self):
        caps = captions_from_measurements(ef=55)
        self.assertEqual(len(caps), 2)
        official = [
            fill_numeric_template(t, 55) for t in ZERO_SHOT_PROMPTS["ejection_fraction"]
        ]
        self.assertEqual(caps, official)
        joined = measurements_to_text(ef=55, include_dilation=False)
        for sentence in official:
            self.assertIn(sentence.strip(), joined)

    def test_edv_maps_to_official_dilation_prompts(self):
        caps = captions_from_measurements(ef=30, edv=260.0)
        dilation = list(ZERO_SHOT_PROMPTS["severe_left_ventricle_dilation"])
        for sentence in dilation:
            self.assertIn(sentence, caps)
        mild = captions_from_measurements(ef=60, edv=160.0)
        self.assertTrue(
            any(s in mild for s in ZERO_SHOT_PROMPTS["mild_left_ventricle_dilation"])
        )

    def test_no_invented_esv_language(self):
        caps = captions_from_measurements(ef=50, esv=40.0, include_dilation=False)
        blob = " ".join(caps).upper()
        self.assertNotIn("END-SYSTOLIC", blob)
        self.assertNotIn("ESV", blob)

    def test_pair_record_requires_ef(self):
        rec = pair_record("Videos/a.avi", ef=62.4, edv=120.0)
        self.assertEqual(rec["image"], "Videos/a.avi")
        self.assertIn("text", rec)
        self.assertEqual(rec["ef"], 62.4)
        with self.assertRaises(ValueError):
            pair_record("Videos/b.avi")


if __name__ == "__main__":
    unittest.main()
