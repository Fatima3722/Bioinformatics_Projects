"""Reproducible educational acetylcholinesterase workflow.

Uses the same pinned DataProfessor archive in every stage. The default baseline
uses its four supplied molecular descriptors. Optional newly generated PaDEL
features must carry workflow record IDs; positional joins are never accepted.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path
from urllib.request import urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from qsar_utils import (
    DataValidationError, adjust_pvalues_bh, align_descriptors,
    prepare_ache_data, safe_mannwhitney,
)

ROOT = Path(__file__).resolve().parent
ARCHIVE_COMMIT = "421ee9d4e2f9c4d7ec4bccbf73c19f773674430a"
ARCHIVE_FILE = "acetylcholinesterase_04_bioactivity_data_3class_pIC50.csv"
ARCHIVE_URL = f"https://raw.githubusercontent.com/dataprofessor/data/{ARCHIVE_COMMIT}/{ARCHIVE_FILE}"
ARCHIVE_SHA256 = "5f12bf531a64ae0f9c764d14c3bbed4620738c6c35d1144c8a647c608e11732f"
FEATURES = ["MW", "LogP", "NumHDonors", "NumHAcceptors"]
SEED = 42


def directories():
    for name in ("data/raw", "outputs", "figures"):
        (ROOT / name).mkdir(parents=True, exist_ok=True)


def acquire_data(source_csv=None):
    """Download or verify the immutable archive, then preserve its provenance.

    ACHE_SOURCE_CSV supplies an offline cache of the exact archived bytes. The
    checksum check prevents an arbitrary local table from inheriting the
    archive's provenance. A fresh ChEMBL export is a different data source.
    """
    directories()
    source_csv = source_csv or os.environ.get("ACHE_SOURCE_CSV")
    path = Path(source_csv) if source_csv else ROOT / "data/raw" / ARCHIVE_FILE
    if not path.exists():
        if source_csv:
            raise FileNotFoundError(path)
        with urlopen(ARCHIVE_URL, timeout=60) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != ARCHIVE_SHA256:
            raise DataValidationError("Downloaded source checksum does not match the pinned archive")
        path.write_bytes(payload)
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != ARCHIVE_SHA256:
        raise DataValidationError("Source checksum does not match the pinned archive")
    records = prepare_ache_data(pd.read_csv(path))
    records = records.drop(columns=[c for c in records if str(c).startswith("Unnamed:")])
    for feature in FEATURES:
        if feature not in records:
            raise DataValidationError(f"Archived source is missing {feature}")
        records[feature] = pd.to_numeric(records[feature], errors="raise")
    if not np.isfinite(records[FEATURES].to_numpy()).all():
        raise DataValidationError("Archived molecular descriptors are nonfinite")
    records.to_csv(ROOT / "outputs/ache_records.csv", index=False)
    provenance = {
        "dataset": "DataProfessor archived educational acetylcholinesterase dataset",
        "source_url": ARCHIVE_URL, "upstream_commit": ARCHIVE_COMMIT,
        "sha256": digest, "rows": len(records),
        "class_counts": records.bioactivity_class.value_counts().to_dict(),
        "environment": {"python": sys.version.split()[0], "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib")
        }},
        "limitations": [
            "The archive omits original target/species, assay, relation, unit and release metadata.",
            "No claim of freshly curated human-only ChEMBL measurements is made.",
            "pIC50 is supplied by the archive; no unavailable raw IC50 measurements are reconstructed.",
        ],
    }
    (ROOT / "outputs/data_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Verified archived AChE records: {len(records):,}")
    print(records.bioactivity_class.value_counts().reindex(["active", "intermediate", "inactive"], fill_value=0))
    return records


def exploratory_analysis(records):
    """Run a fixed four-descriptor family, with BH adjustment and count gates."""
    directories()
    tests = pd.DataFrame([safe_mannwhitney(records, feature) for feature in FEATURES])
    tests["p_adjusted_bh"] = adjust_pvalues_bh(tests.p_value)
    tests["adjusted_interpretation"] = [
        "Not interpreted: invalid or unavailable test" if not np.isfinite(q)
        else "Exploratory evidence against the null after BH adjustment" if q <= 0.05
        else "Insufficient evidence to reject the null after BH adjustment"
        for q in tests.p_adjusted_bh
    ]
    tests.to_csv(ROOT / "outputs/descriptor_comparisons.csv", index=False)
    counts = records.bioactivity_class.value_counts().reindex(["active", "intermediate", "inactive"], fill_value=0)
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    ax.bar(counts.index, counts.values, color=["#2c7a7b", "#b7791f", "#4a5568"])
    ax.set(ylabel="Compounds", title="Archived AChE dataset: fixed activity thresholds")
    fig.tight_layout(); fig.savefig(ROOT / "figures/activity_class_counts.png", dpi=170); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7))
    for ax, feature in zip(axes.ravel(), FEATURES):
        groups = [records.loc[records.bioactivity_class.eq(label), feature].to_numpy() for label in ("active", "inactive")]
        ax.boxplot(groups, tick_labels=["Active", "Inactive"], showfliers=False)
        ax.set(ylabel=feature)
    fig.suptitle("Molecular descriptors by activity class", fontsize=14)
    fig.tight_layout(); fig.savefig(ROOT / "figures/descriptor_distributions.png", dpi=170); plt.close(fig)
    print(tests[["descriptor", "n_active", "n_inactive", "status", "p_value", "p_adjusted_bh"]].to_string(index=False))
    print("Exploratory comparisons only. Classes are defined from pIC50, so pIC50 is not tested between them.")
    return tests


def prepare_descriptors(records, padel_csv=None):
    """Export PaDEL inputs and build an ID-aligned modelling table.

    The runnable default uses the archive's four supplied descriptors. Set
    ACHE_PADEL_CSV to newly computed PaDEL output to use those features instead.
    The PaDEL Name field must contain the exported record_id, not row numbers.
    """
    directories()
    records[["canonical_smiles", "record_id"]].to_csv(
        ROOT / "outputs/ache_molecules.smi", sep="\t", index=False, header=False,
    )
    metadata = records[["record_id", "molecule_chembl_id", "canonical_smiles", "pIC50", "bioactivity_class"]].copy()
    padel_csv = padel_csv or os.environ.get("ACHE_PADEL_CSV")
    if padel_csv:
        descriptors = pd.read_csv(padel_csv)
        table = align_descriptors(metadata, descriptors, key="record_id", descriptor_key="Name")
        features = [c for c in descriptors if c != "Name"]
        representation = "ID-aligned PaDEL descriptors generated from the exported AChE records"
    else:
        descriptors = records[["record_id", *FEATURES]].copy()
        table = align_descriptors(metadata, descriptors, key="record_id")
        features = FEATURES.copy()
        representation = "Four molecular descriptors supplied in the archived dataset"
    table.to_csv(ROOT / "outputs/ache_model_dataset.csv", index=False)
    specification = {"representation": representation, "features": features, "count": len(features)}
    (ROOT / "outputs/feature_columns.json").write_text(json.dumps(specification, indent=2) + "\n")
    print(f"Aligned records: {len(table):,}; predictors: {len(features)}")
    print(representation)
    return table, features


def split_data(table, features):
    """One fixed molecule-group split, shared by the two evaluation notebooks."""
    if not table.record_id.is_unique:
        raise DataValidationError("Model records must have unique record IDs")
    if not features or set(features) & {"pIC50", "bioactivity_class", "class", "record_id", "molecule_chembl_id", "canonical_smiles"}:
        raise DataValidationError("Predictors must exclude outcomes, labels and identifiers")
    X = table[features].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(X.to_numpy()).all():
        raise DataValidationError("Model features must be finite")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train, test = next(splitter.split(X, table.pIC50, groups=table.molecule_chembl_id))
    if set(table.iloc[train].molecule_chembl_id) & set(table.iloc[test].molecule_chembl_id):
        raise DataValidationError("A molecule occurs in both train and test sets")
    # This archive has unique SMILES. Guard against exact-structure leakage if
    # the source is later replaced by a table with different identifiers.
    if set(table.iloc[train].canonical_smiles) & set(table.iloc[test].canonical_smiles):
        raise DataValidationError("An exact SMILES structure occurs in both splits; revise grouping")
    manifest = table[["record_id", "molecule_chembl_id"]].copy()
    manifest["split"] = "test"; manifest.iloc[train, manifest.columns.get_loc("split")] = "train"
    manifest.to_csv(ROOT / "outputs/split_manifest.csv", index=False)
    return X, table.pIC50.astype(float), train, test


def estimator_pipeline(estimator, scale=False):
    steps = [("imputer", SimpleImputer(strategy="median")), ("variance", VarianceThreshold())]
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline([*steps, ("regressor", estimator)])


def random_forest_baseline(table, features):
    """Evaluate a preselected forest and a training-mean baseline on held-out molecules."""
    X, y, train, test = split_data(table, features)
    models = {
        "Training-mean baseline": DummyRegressor(strategy="mean"),
        "Random forest": estimator_pipeline(RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=2)),
    }
    rows, predictions = [], table.iloc[test][["record_id", "molecule_chembl_id"]].copy()
    predictions["observed_pIC50"] = y.iloc[test].to_numpy()
    for name, model in models.items():
        model.fit(X.iloc[train], y.iloc[train])
        predicted = model.predict(X.iloc[test])
        rows.append({"model": name, "n_train": len(train), "n_test": len(test),
                     "MAE": mean_absolute_error(y.iloc[test], predicted),
                     "RMSE": float(np.sqrt(mean_squared_error(y.iloc[test], predicted))),
                     "R2": r2_score(y.iloc[test], predicted)})
        predictions[name] = predicted
    metrics = pd.DataFrame(rows)
    metrics.to_csv(ROOT / "outputs/random_forest_holdout_metrics.csv", index=False)
    predictions.to_csv(ROOT / "outputs/holdout_predictions.csv", index=False)
    observed = y.iloc[test].to_numpy(); predicted = predictions["Random forest"].to_numpy()
    lo, hi = min(observed.min(), predicted.min()), max(observed.max(), predicted.max())
    fig, ax = plt.subplots(figsize=(6.1, 5.1))
    ax.scatter(observed, predicted, s=12, alpha=0.35, color="#276749")
    ax.plot([lo, hi], [lo, hi], color="#4a5568", linestyle="--", label="Identity")
    ax.set(xlabel="Observed pIC50", ylabel="Predicted pIC50", title="Preselected random forest: held-out molecules")
    ax.legend(); fig.tight_layout(); fig.savefig(ROOT / "figures/random_forest_holdout.png", dpi=170); plt.close(fig)
    print(metrics.to_string(index=False))
    print("Educational random-molecule holdout; not scaffold or external validation.")
    return metrics


def compare_regressors(table, features):
    """Compare fixed models by group cross-validation within training data only.

    The held-out set is deliberately not scored or used to rank these models.
    This table describes CV validation performance, never training-fit scores.
    """
    X, y, train, _ = split_data(table, features)
    X_train, y_train = X.iloc[train], y.iloc[train]
    groups = table.iloc[train].molecule_chembl_id
    if groups.nunique() < 5:
        raise DataValidationError("Five-fold group cross-validation needs at least five training groups")
    cv = GroupKFold(n_splits=5)
    models = {
        "Training-mean baseline": DummyRegressor(strategy="mean"),
        "Ridge": estimator_pipeline(Ridge(alpha=10), scale=True),
        "Random forest": estimator_pipeline(RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=1)),
        "Extra trees": estimator_pipeline(ExtraTreesRegressor(n_estimators=100, random_state=SEED, n_jobs=1)),
    }
    rows = []
    for name, model in models.items():
        scores = cross_validate(model, X_train, y_train, groups=groups, cv=cv,
                                scoring={"MAE": "neg_mean_absolute_error", "RMSE": "neg_root_mean_squared_error", "R2": "r2"},
                                n_jobs=1, error_score="raise", return_train_score=False)
        row = {"model": name, "folds": 5}
        for metric in ("MAE", "RMSE", "R2"):
            values = scores["test_" + metric]
            if metric != "R2": values = -values
            row[metric + "_mean"] = float(np.mean(values))
            row[metric + "_sd"] = float(np.std(values, ddof=1))
        rows.append(row)
    comparison = pd.DataFrame(rows).sort_values("MAE_mean").reset_index(drop=True)
    comparison.to_csv(ROOT / "outputs/training_group_cv_metrics.csv", index=False)
    print(comparison.to_string(index=False))
    print("Model ranking uses training-set group CV only. The held-out set is not used to rank models.")
    return comparison
