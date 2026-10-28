"""Focused invariant tests; runnable with unittest or pytest."""

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qsar_utils import (
    DataValidationError,
    adjust_pvalues_bh,
    align_descriptors,
    prepare_ache_data,
    safe_mannwhitney,
)


def example_records():
    return pd.DataFrame(
        {
            "molecule_chembl_id": ["CHEMBL1", "CHEMBL2", "CHEMBL3", "CHEMBL4"],
            "canonical_smiles": ["CC", "CCC", "CCCC", "CCO"],
            "pIC50": [6.0, 5.0, 5.5, 7.0],
            "class": ["inactive", "active", "active", "inactive"],
        }
    )


class PrepareTests(unittest.TestCase):
    def test_recalculates_labels_and_retains_source(self):
        raw = example_records()
        before = raw.copy(deep=True)
        clean = prepare_ache_data(raw)
        self.assertEqual(clean.bioactivity_class.tolist(), ["active", "inactive", "intermediate", "active"])
        self.assertEqual(clean.source_bioactivity_class.tolist(), raw["class"].tolist())
        self.assertNotIn("class", clean)
        pd.testing.assert_frame_equal(raw, before)

    def test_preserves_valid_existing_record_ids(self):
        raw = example_records().assign(record_id=["r1", "r2", "r3", "r4"])
        self.assertEqual(prepare_ache_data(raw).record_id.tolist(), raw.record_id.tolist())

    def test_generated_ids_stable_after_row_reordering(self):
        raw = example_records()
        first = prepare_ache_data(raw).set_index("molecule_chembl_id").record_id.sort_index()
        second = prepare_ache_data(raw.sample(frac=1, random_state=4).reset_index(drop=True))
        second = second.set_index("molecule_chembl_id").record_id.sort_index()
        pd.testing.assert_series_equal(first, second)

    def test_duplicate_molecule_records_preserved_with_unique_ids(self):
        raw = pd.concat([example_records(), example_records().iloc[[0]]], ignore_index=True)
        clean = prepare_ache_data(raw)
        self.assertEqual(len(clean), 5)
        self.assertTrue(clean.record_id.is_unique)
        self.assertEqual(clean.molecule_chembl_id.tolist().count("CHEMBL1"), 2)

    def test_prepare_is_idempotent(self):
        clean = prepare_ache_data(example_records())
        pd.testing.assert_frame_equal(clean, prepare_ache_data(clean))

    def test_rejects_invalid_data_without_silent_row_dropping(self):
        for bad in (np.nan, np.inf, -np.inf, "not-a-number"):
            raw = example_records().astype({"pIC50": object})
            raw.loc[0, "pIC50"] = bad
            with self.subTest(value=bad), self.assertRaises(DataValidationError):
                prepare_ache_data(raw)

    def test_rejects_missing_column_or_empty_data(self):
        with self.assertRaises(DataValidationError):
            prepare_ache_data(example_records().drop(columns="canonical_smiles"))
        with self.assertRaises(DataValidationError):
            prepare_ache_data(example_records().iloc[0:0])

    def test_rejects_ambiguous_record_ids(self):
        for ids in (["r1", "r1", "r3", "r4"], ["r1", None, "r3", "r4"], ["r1", "", "r3", "r4"]):
            with self.subTest(ids=ids), self.assertRaises(DataValidationError):
                prepare_ache_data(example_records().assign(record_id=ids))


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.records = prepare_ache_data(example_records())
        self.descriptors = pd.DataFrame({"record_id": self.records.record_id, "fingerprint": [10, 20, 30, 40]})

    def test_shuffled_rows_align_by_identity_and_inputs_unchanged(self):
        shuffled = self.descriptors.sample(frac=1, random_state=8)
        left_before = self.records.copy(deep=True)
        right_before = shuffled.copy(deep=True)
        result = align_descriptors(self.records, shuffled)
        self.assertEqual(result.fingerprint.tolist(), [10, 20, 30, 40])
        self.assertEqual(result.record_id.tolist(), self.records.record_id.tolist())
        pd.testing.assert_frame_equal(left_before, self.records)
        pd.testing.assert_frame_equal(right_before, shuffled)

    def test_padel_name_column_with_repeated_molecule_ids(self):
        raw = example_records()
        raw.loc[1, "molecule_chembl_id"] = "CHEMBL1"
        records = prepare_ache_data(raw)
        padel = pd.DataFrame({"Name": records.record_id, "fp": [1, 2, 3, 4]}).iloc[::-1]
        merged = align_descriptors(records, padel, key="record_id", descriptor_key="Name")
        self.assertEqual(merged.fp.tolist(), [1, 2, 3, 4])

    def test_fallback_molecule_key_is_allowed_when_unique(self):
        descriptors = pd.DataFrame({"molecule_chembl_id": self.records.molecule_chembl_id, "fp": [1, 2, 3, 4]}).iloc[::-1]
        self.assertEqual(align_descriptors(self.records, descriptors).fp.tolist(), [1, 2, 3, 4])

    def test_fallback_rejects_duplicate_molecules(self):
        records = self.records.copy()
        records.loc[1, "molecule_chembl_id"] = records.loc[0, "molecule_chembl_id"]
        descriptors = pd.DataFrame({"molecule_chembl_id": records.molecule_chembl_id, "fp": [1, 2, 3, 4]})
        with self.assertRaisesRegex(DataValidationError, "duplicate"):
            align_descriptors(records, descriptors)

    def test_duplicate_descriptor_record_key_rejected(self):
        descriptors = pd.concat([self.descriptors, self.descriptors.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(DataValidationError, "duplicate"):
            align_descriptors(self.records, descriptors)

    def test_missing_or_extra_rows_rejected(self):
        missing = self.descriptors.iloc[:-1]
        extra = pd.concat([self.descriptors, pd.DataFrame({"record_id": ["extra"], "fingerprint": [99]})])
        changed = self.descriptors.copy()
        changed.loc[0, "record_id"] = "wrong-identity"
        for descriptors in (missing, extra, changed):
            with self.subTest(keys=descriptors.record_id.tolist()), self.assertRaisesRegex(DataValidationError, "keys do not match"):
                align_descriptors(self.records, descriptors)

    def test_missing_feature_values_and_column_collision_rejected(self):
        bad = self.descriptors.copy()
        bad.loc[0, "fingerprint"] = np.nan
        with self.assertRaisesRegex(DataValidationError, "nonfinite"):
            align_descriptors(self.records, bad)
        with self.assertRaisesRegex(DataValidationError, "overwrite"):
            align_descriptors(self.records, self.descriptors.rename(columns={"fingerprint": "pIC50"}))


class MannWhitneyTests(unittest.TestCase):
    def test_absent_active_group_is_not_tested(self):
        data = pd.DataFrame({"bioactivity_class": ["inactive"] * 3, "MW": [200, 300, 400]})
        with patch("qsar_utils.stats.mannwhitneyu") as test:
            result = safe_mannwhitney(data, "MW")
            test.assert_not_called()
        self.assertEqual(result["n_active"], 0)
        self.assertEqual(result["n_inactive"], 3)
        self.assertEqual(result["status"], "not_tested")
        self.assertTrue(np.isnan(result["p_value"]))

    def test_filters_nonfinite_values_and_reports_exclusions(self):
        data = pd.DataFrame({"bioactivity_class": ["active"] * 3 + ["inactive"] * 3,
                             "MW": [100, 120, np.nan, 200, 220, np.inf]})
        before = data.copy(deep=True)
        result = safe_mannwhitney(data, "MW")
        self.assertEqual((result["n_active"], result["n_inactive"]), (2, 2))
        self.assertEqual((result["n_active_excluded"], result["n_inactive_excluded"]), (1, 1))
        self.assertEqual(result["status"], "ok")
        self.assertTrue(np.isfinite(result["p_value"]))
        pd.testing.assert_frame_equal(data, before)

    def test_nan_or_invalid_test_result_never_rejects_null(self):
        data = pd.DataFrame({"bioactivity_class": ["active", "inactive"], "MW": [1, 2]})
        for statistic, pvalue in ((np.nan, np.nan), (1, np.nan), (np.inf, .01), (1, 1.2)):
            with self.subTest(pvalue=pvalue), patch("qsar_utils.stats.mannwhitneyu", return_value=SimpleNamespace(statistic=statistic, pvalue=pvalue)):
                result = safe_mannwhitney(data, "MW")
                self.assertEqual(result["status"], "invalid_result")
                self.assertNotIn("Evidence against", result["interpretation"])

    def test_refuses_circular_pic50_test(self):
        with self.assertRaisesRegex(DataValidationError, "classes defined"):
            safe_mannwhitney(prepare_ache_data(example_records()), "pIC50")

    def test_fail_to_reject_is_not_called_same_distribution(self):
        data = pd.DataFrame({"bioactivity_class": ["active", "inactive"], "MW": [1, 1]})
        result = safe_mannwhitney(data, "MW")
        self.assertIn("does not establish equivalence", result["interpretation"])
        self.assertNotIn("Same distribution", result["interpretation"])


class MultipleTestingTests(unittest.TestCase):
    def test_bh_matches_known_values_and_preserves_missing_positions(self):
        raw = [.01, np.nan, .04, .03, .002]
        np.testing.assert_allclose(adjust_pvalues_bh(raw), [.02, np.nan, .04, .04, .008], equal_nan=True)

    def test_all_missing_and_empty_values_are_safe(self):
        self.assertEqual(adjust_pvalues_bh([]).shape, (0,))
        self.assertTrue(np.isnan(adjust_pvalues_bh([np.nan, np.nan])).all())

    def test_invalid_pvalues_are_not_silently_adjusted(self):
        for raw in ([np.inf], [-.01], [1.01], [[.01, .02]]):
            with self.subTest(values=raw), self.assertRaises(DataValidationError):
                adjust_pvalues_bh(raw)


if __name__ == "__main__":
    unittest.main()
