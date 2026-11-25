"""Self-contained offline unit tests for calibration, cohort summaries, and paired bootstrap CIs."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from committee_ph2 import read_scores
from final_benchmark_intervals import paired_results
from final_external_benchmark import (
    CUTOFFS,
    PRIMARY,
    _ph2_labels,
    calibrated_blend_cutoff,
    shared_cutoffs,
    summarize_all_datasets,
)


class BenchmarkTests(unittest.TestCase):
    def test_ph2_labels_parses_reference(self):
        ref = "\n".join([
            "|| Name || Histological Diagnosis || Clinical Diagnosis || Other ||",
            "|| IMD001 || 0 || 0 || x ||",
            "|| IMD002 || 1 || 1 || x ||",
            "|| IMD003 || 2 || 2 || x ||",
        ])
        parsed = _ph2_labels(ref)
        self.assertEqual(parsed, {"IMD001": 0, "IMD002": 1, "IMD003": 2})

    def test_shared_calibration_and_tie_handling(self):
        rng = np.random.default_rng(11)
        n_cal = 71
        cal = pd.DataFrame({
            "image_id": [f"CAL_{i}" for i in range(n_cal)],
            "patient_id": [f"P_{i}" for i in range(n_cal)],
            "split": "calibration",
            "cohort": "nevus",
            "retained": rng.normal(-5.0, 1.0, n_cal),
            "contrast_gamma": rng.normal(-3.0, 1.0, n_cal),
            "benign_mimics": rng.normal(-3.5, 1.0, n_cal),
            "siim_retrained": rng.normal(0.0, 0.5, n_cal),
            "balanced_mel_nv": rng.normal(-3.0, 1.0, n_cal),
        })
        cal["blend_60_40"] = 0.6 * cal.retained + 0.4 * cal.balanced_mel_nv
        cal["tied"] = np.r_[np.ones(8), np.zeros(n_cal - 8)]
        cutoffs = shared_cutoffs(cal, [*PRIMARY, "tied"])
        for model, cutoff in cutoffs.items():
            self.assertLessEqual(cal[model].ge(cutoff).sum(), 7)
        self.assertEqual(cal.tied.ge(cutoffs["tied"]).sum(), 0)
        with self.assertRaises(ValueError):
            shared_cutoffs(cal.iloc[:-1], PRIMARY)

    def test_paired_auc_and_decision_counts(self):
        df = pd.DataFrame({
            "image_id": list("abcdef"),
            "patient_id": [1, 2, 1, 3, 4, 5],
            "blend_60_40": [0.5, 0.3, 0.1, 0.8, 0.6, 0.7],
            "comparator": [0.4, 0.5, 0.4, 0.7, 0.7, 0.7],
        })
        y = np.array([0, 0, 0, 1, 1, 1])
        res = paired_results(df, y, {"blend_60_40": 0.5, "comparator": 0.5}, "example", draws=100)
        self.assertAlmostEqual(
            res.iloc[2].blend_minus_comparator,
            roc_auc_score(y, df.blend_60_40) - roc_auc_score(y, df.comparator),
        )
        self.assertEqual(res.iloc[1].new_alerts, 1)
        self.assertEqual(res.iloc[1].cleared_alerts, 1)
        self.assertTrue((res.cluster == "patient_id").all())

    def test_committee_score_alignment(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "softmax_cm.csv"
            path.write_text("name,pred,true\nb,0.8,0\na,0.2,0\n", encoding="utf-8")
            self.assertEqual(read_scores(path, ["a", "b"]).tolist(), [0.2, 0.8])
            with self.assertRaises(ValueError):
                read_scores(path, ["a", "c"])

    def test_calibrated_blend_cutoff_and_summarize_all_datasets(self):
        val_df = pd.DataFrame({
            "role": "calibration",
            "label": [1] * 20 + [0] * 534,
            "retained": np.concatenate([np.linspace(-2.0, -0.5, 20), np.linspace(-7.0, -4.0, 534)]),
            "balanced_mel_nv": np.concatenate([np.linspace(-1.5, -0.2, 20), np.linspace(-5.0, -2.5, 534)]),
        })
        cut = calibrated_blend_cutoff(val_df, 0.6)
        self.assertTrue(np.isfinite(cut))
        cutoffs = {**CUTOFFS, "blend_60_40": cut}

        rng = np.random.default_rng(7)

        def _scores(n, pos_offset=0.0):
            return {
                "retained": rng.normal(-5.0 + pos_offset, 1.0, n),
                "contrast_gamma": rng.normal(-3.5 + pos_offset, 1.0, n),
                "benign_mimics": rng.normal(-4.0 + pos_offset, 1.0, n),
                "siim_retrained": rng.normal(0.0 + pos_offset * 0.2, 0.5, n),
                "balanced_mel_nv": rng.normal(-3.8 + pos_offset, 1.0, n),
            }

        isic_df = pd.DataFrame({
            "image_id": [f"ISIC_{i}" for i in range(80)],
            "label": [1] * 20 + [0] * 60,
            "in_labelled_nevi": [False] * 20 + [True] * 40 + [False] * 20,
            "in_unspecified_benign": [True] * 20 + [False] * 40 + [True] * 20,
            "in_low_contrast": [True] * 10 + [False] * 10 + [True] * 15 + [False] * 45,
            **_scores(80),
        })
        isic_df["blend_60_40"] = 0.6 * isic_df["retained"] + 0.4 * isic_df["balanced_mel_nv"]

        derm_df = pd.DataFrame({
            "image_id": [f"DERM_{i}" for i in range(60)],
            "patient_id": [f"DP_{i}" for i in range(60)],
            "cohort": ["melanoma"] * 15 + ["ordinary_nevi"] * 30 + ["seborrheic_keratosis"] * 15,
            **_scores(60),
        })
        derm_df["blend_60_40"] = 0.6 * derm_df["retained"] + 0.4 * derm_df["balanced_mel_nv"]

        prove_df = pd.DataFrame({
            "image_id": [f"PROVE_{i}" for i in range(131)],
            "patient_id": [f"PP_{i}" for i in range(131)],
            "split": ["calibration"] * 71 + ["test"] * 60,
            "cohort": ["nevus"] * 71 + ["melanoma"] * 20 + ["nevus"] * 30 + ["bcc"] * 10,
            **_scores(131),
        })
        prove_df["blend_60_40"] = 0.6 * prove_df["retained"] + 0.4 * prove_df["balanced_mel_nv"]

        ph2_df = pd.DataFrame({
            "image_id": [f"IMD{i:03d}" for i in range(40)],
            "diagnosis": ["melanoma"] * 10 + ["nevus"] * 30,
            **_scores(40),
        })
        ph2_df["blend_60_40"] = 0.6 * ph2_df["retained"] + 0.4 * ph2_df["balanced_mel_nv"]

        mn_df, mimic_df = summarize_all_datasets(cutoffs, isic_df, derm_df, prove_df, ph2_df)
        self.assertEqual(set(mn_df["dataset"]), {"ISIC 2020", "DERM12345", "PROVe-AI", "PH2"})
        self.assertEqual(set(mimic_df["dataset"]), {"ISIC 2020", "DERM12345", "PROVe-AI"})


if __name__ == "__main__":
    unittest.main()
