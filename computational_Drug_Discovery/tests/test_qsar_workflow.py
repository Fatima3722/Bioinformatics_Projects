import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

import qsar_workflow as workflow
from qsar_utils import DataValidationError, prepare_ache_data


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root_patch = patch.object(workflow, "ROOT", Path(self.temp.name))
        self.root_patch.start(); self.addCleanup(self.root_patch.stop)
        workflow.directories()
        self.records = prepare_ache_data(pd.DataFrame({
            "molecule_chembl_id": [f"CHEMBL{i // 2 + 1}" for i in range(24)],
            "canonical_smiles": [f"molecule_{i // 2}" for i in range(24)],
            "pIC50": np.linspace(4, 8, 24),
            "MW": np.arange(24) + 100, "LogP": np.arange(24) / 10,
            "NumHDonors": np.arange(24) % 3, "NumHAcceptors": np.arange(24) % 4,
        }))

    def test_archive_cache_checksum_is_enforced(self):
        path = Path(self.temp.name) / "wrong.csv"
        path.write_text("molecule_chembl_id,pIC50\nCHEMBL1,6\n")
        with self.assertRaisesRegex(DataValidationError, "checksum"):
            workflow.acquire_data(path)

    def test_default_descriptor_table_preserves_activity_identity(self):
        with patch.dict(os.environ, {"ACHE_PADEL_CSV": ""}):
            table, features = workflow.prepare_descriptors(self.records)
        self.assertEqual(features, workflow.FEATURES)
        pd.testing.assert_series_equal(table.pIC50, self.records.pIC50)
        pd.testing.assert_series_equal(table.MW, self.records.MW)
        self.assertTrue(table.record_id.is_unique)
        specification = json.loads((workflow.ROOT / "outputs/feature_columns.json").read_text())
        self.assertEqual(specification["count"], 4)

    def test_optional_padel_rows_can_be_reordered_without_corrupting_labels(self):
        descriptors = pd.DataFrame({"Name": self.records.record_id, "PubchemFP0": np.arange(24) % 2})
        path = Path(self.temp.name) / "padel.csv"
        descriptors.sample(frac=1, random_state=7).to_csv(path, index=False)
        table, features = workflow.prepare_descriptors(self.records, path)
        self.assertEqual(features, ["PubchemFP0"])
        np.testing.assert_array_equal(table.PubchemFP0, descriptors.PubchemFP0)
        pd.testing.assert_series_equal(table.pIC50, self.records.pIC50)

    def test_split_is_reproducible_and_molecules_are_disjoint(self):
        X, y, train, test = workflow.split_data(self.records, workflow.FEATURES)
        _, _, train2, test2 = workflow.split_data(self.records, workflow.FEATURES)
        np.testing.assert_array_equal(train, train2); np.testing.assert_array_equal(test, test2)
        self.assertFalse(set(self.records.iloc[train].molecule_chembl_id) & set(self.records.iloc[test].molecule_chembl_id))
        self.assertEqual(set(train) | set(test), set(range(len(self.records))))

    def test_outcomes_and_identifiers_cannot_be_predictors(self):
        for forbidden in ("pIC50", "bioactivity_class", "record_id", "molecule_chembl_id"):
            with self.subTest(forbidden=forbidden), self.assertRaises(DataValidationError):
                workflow.split_data(self.records, ["MW", forbidden])

    def test_exact_structure_leakage_is_rejected(self):
        records = self.records.copy()
        records["canonical_smiles"] = "same_structure"
        with self.assertRaisesRegex(DataValidationError, "SMILES"):
            workflow.split_data(records, workflow.FEATURES)

    def test_preprocessing_is_fitted_from_training_data(self):
        train = pd.DataFrame({"variable": [0., 1., 2.], "test_only_variation": [0., 0., 0.]})
        test = pd.DataFrame({"variable": [100.], "test_only_variation": [999.]})
        model = workflow.estimator_pipeline(Ridge())
        model.fit(train, [0., 1., 2.])
        np.testing.assert_array_equal(model.named_steps["variance"].get_support(), [True, False])
        np.testing.assert_allclose(model.named_steps["imputer"].statistics_, [1., 0.])
        self.assertEqual(len(model.predict(test)), 1)


if __name__ == "__main__":
    unittest.main()
