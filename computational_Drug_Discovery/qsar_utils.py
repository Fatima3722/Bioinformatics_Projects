"""Validation and analysis helpers for the acetylcholinesterase learning workflow.

These helpers validate table structure and analysis invariants; they do not
establish the biological provenance of the input or validate SMILES chemically.
No RDKit installation is required to import this module.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy import stats


class DataValidationError(ValueError):
    """An input violates an invariant needed for a defensible analysis."""


_REQUIRED_COLUMNS = ("molecule_chembl_id", "canonical_smiles", "pIC50")


def _check_columns(frame: pd.DataFrame, name: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    if not frame.columns.is_unique:
        raise DataValidationError(f"{name} contains duplicate column names")


def _key_values(frame: pd.DataFrame, key: str, name: str) -> pd.Series:
    if key not in frame.columns:
        raise DataValidationError(f"{name} is missing key column {key!r}")
    values = frame[key]
    if values.isna().any():
        raise DataValidationError(f"{name}.{key} contains missing keys")
    if not values.map(lambda value: isinstance(value, str)).all():
        raise DataValidationError(f"{name}.{key} must contain string identifiers")
    if values.str.strip().eq("").any():
        raise DataValidationError(f"{name}.{key} contains empty keys")
    if values.duplicated().any():
        examples = values[values.duplicated(keep=False)].unique()[:3].tolist()
        raise DataValidationError(
            f"{name}.{key} contains ambiguous duplicate keys: {examples}. "
            "Use unique record_id values for repeated molecule measurements."
        )
    return values


def prepare_ache_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a validated copy with stable record IDs and recalculated classes.

    Required columns are ``molecule_chembl_id``, ``canonical_smiles`` and
    finite numeric ``pIC50``. An input ``class`` or ``bioactivity_class`` is
    retained for audit in ``source_bioactivity_class``; the canonical output
    label is always recomputed: >= 6 active, <= 5 inactive, otherwise
    intermediate. Input labels never determine the new classification.

    Repeated molecule measurements are preserved. Existing unique record IDs
    are retained; otherwise deterministic content hashes and duplicate-instance
    suffixes are generated. Keep these IDs in every subsequent export: they are
    workflow row identifiers, not ChEMBL activity identifiers. Identical input
    rows are indistinguishable without an external activity identifier.
    """
    _check_columns(frame, "data")
    missing = [column for column in _REQUIRED_COLUMNS if column not in frame]
    if missing:
        raise DataValidationError(f"data is missing required columns: {missing}")
    if frame.empty:
        raise DataValidationError("data contains no records")
    out = frame.copy(deep=True)
    for column in ("molecule_chembl_id", "canonical_smiles"):
        if out[column].isna().any() or not out[column].map(
            lambda value: isinstance(value, str) and bool(value.strip())
        ).all():
            raise DataValidationError(f"{column} must contain nonempty strings")
        out[column] = out[column].str.strip()
    if not out["molecule_chembl_id"].str.fullmatch(r"CHEMBL\d+").all():
        raise DataValidationError("molecule_chembl_id must contain CHEMBL identifiers")
    try:
        values = pd.to_numeric(out["pIC50"], errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise DataValidationError("pIC50 must contain numeric values") from exc
    if not np.isfinite(values.to_numpy()).all():
        raise DataValidationError("pIC50 contains missing or nonfinite values")
    out["pIC50"] = values

    if "source_bioactivity_class" not in out:
        if "bioactivity_class" in out:
            out["source_bioactivity_class"] = out["bioactivity_class"].copy()
        elif "class" in out:
            out["source_bioactivity_class"] = out["class"].copy()
        else:
            out["source_bioactivity_class"] = pd.NA
    if "class" in out:
        if "source_class" not in out:
            out["source_class"] = out["class"].copy()
        out = out.drop(columns="class")
    out["bioactivity_class"] = np.select(
        [values.ge(6.0), values.le(5.0)],
        ["active", "inactive"],
        default="intermediate",
    )

    if "record_id" not in out:
        payloads = [
            "\x1f".join((molecule, smiles, format(pic50, ".17g")))
            for molecule, smiles, pic50 in zip(
                out["molecule_chembl_id"], out["canonical_smiles"], values
            )
        ]
        hashes = pd.Series(
            [hashlib.sha256(item.encode("utf-8")).hexdigest()[:24] for item in payloads],
            index=out.index,
        )
        occurrences = hashes.groupby(hashes, sort=False).cumcount()
        out["record_id"] = (
            "ache_" + hashes + "_" + occurrences.astype(str)
        ).to_numpy()
    _key_values(out, "record_id", "data")
    return out


def align_descriptors(
    records: pd.DataFrame,
    descriptors: pd.DataFrame,
    *,
    key: str | None = None,
    descriptor_key: str | None = None,
) -> pd.DataFrame:
    """Join descriptor rows to records by identity, never by row position.

    If both inputs have ``record_id``, that key is preferred. Otherwise the
    default fallback is ``molecule_chembl_id``, which must be unique in both
    tables. Repeated molecule IDs are never expanded through a many-to-many
    join. Missing, extra, duplicate or empty descriptor keys are rejected.

    For PaDEL output with workflow record IDs stored in its ``Name`` column,
    call ``align_descriptors(records, padel, key='record_id',
    descriptor_key='Name')``. Descriptor feature columns must be finite numeric
    values and must not overwrite record metadata. The records' order and
    index are preserved; neither input is changed.
    """
    _check_columns(records, "records")
    _check_columns(descriptors, "descriptors")
    if records.empty or descriptors.empty:
        raise DataValidationError("records and descriptors must be nonempty")
    if key is None:
        key = (
            "record_id"
            if "record_id" in records and "record_id" in descriptors
            else "molecule_chembl_id"
        )
    if descriptor_key is None:
        descriptor_key = key
    left_keys = _key_values(records, key, "records")
    right_keys = _key_values(descriptors, descriptor_key, "descriptors")
    missing = set(left_keys) - set(right_keys)
    extra = set(right_keys) - set(left_keys)
    if missing or extra:
        raise DataValidationError(
            "Descriptor keys do not match the records: "
            f"{len(missing)} missing, {len(extra)} extra. "
            "No rows were dropped or matched by position."
        )
    features = descriptors.drop(columns=descriptor_key).copy(deep=True)
    if features.empty:
        raise DataValidationError("descriptors contains no feature columns")
    overlap = set(features.columns) & set(records.columns)
    if overlap:
        raise DataValidationError(f"Descriptor columns overwrite record columns: {sorted(overlap)}")
    try:
        features = features.apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise DataValidationError("Descriptor feature columns must be numeric") from exc
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise DataValidationError("Descriptor features contain missing or nonfinite values")
    features.index = right_keys.to_numpy()
    ordered = features.loc[left_keys.to_numpy()].copy()
    ordered.index = records.index.copy()
    return pd.concat([records.copy(deep=True), ordered], axis=1)


def safe_mannwhitney(
    data: pd.DataFrame,
    descriptor: str,
    *,
    class_col: str = "bioactivity_class",
    alpha: float = 0.05,
) -> dict[str, object]:
    """Compare finite descriptor observations in active and inactive groups.

    Returns counts and an explicit status. Empty groups and nonfinite test
    results never generate a rejection decision. Missing/nonfinite descriptor
    values are excluded separately per group and their counts are reported.
    The computational nonempty-group gate is not evidence of adequate sample
    size, independence, assay comparability or statistical power.

    Do not test pIC50 between classes defined using pIC50: that comparison is
    circular. This function rejects that endpoint explicitly. Returned raw
    p-values can be adjusted across a prespecified descriptor family with
    ``adjust_pvalues_bh`` before drawing multiplicity-aware conclusions.
    """
    _check_columns(data, "data")
    if descriptor.casefold() == "pic50":
        raise DataValidationError("Do not test pIC50 between classes defined from pIC50")
    if descriptor not in data or class_col not in data:
        raise DataValidationError(f"data must contain {descriptor!r} and {class_col!r}")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise DataValidationError("alpha must be finite and strictly between 0 and 1")
    labels = data[class_col]
    invalid_labels = set(labels.dropna()) - {"active", "inactive", "intermediate"}
    if invalid_labels:
        raise DataValidationError(f"Unexpected bioactivity labels: {sorted(invalid_labels)}")
    try:
        numeric = pd.to_numeric(data[descriptor], errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"{descriptor} must contain numeric values") from exc
    finite = np.isfinite(numeric.to_numpy())
    active_mask = labels.eq("active").fillna(False).to_numpy(dtype=bool)
    inactive_mask = labels.eq("inactive").fillna(False).to_numpy(dtype=bool)
    active = numeric.to_numpy()[active_mask & finite]
    inactive = numeric.to_numpy()[inactive_mask & finite]
    result: dict[str, object] = {
        "descriptor": descriptor,
        "n_active": len(active),
        "n_inactive": len(inactive),
        "n_active_excluded": int(np.sum(active_mask & ~finite)),
        "n_inactive_excluded": int(np.sum(inactive_mask & ~finite)),
        "statistic": np.nan,
        "p_value": np.nan,
        "status": "not_tested",
        "interpretation": "",
    }
    if not len(active) or not len(inactive):
        result["interpretation"] = "Not tested: at least one group has no finite observations."
        return result
    test = stats.mannwhitneyu(active, inactive, alternative="two-sided", method="auto")
    if not np.isfinite(test.statistic) or not np.isfinite(test.pvalue):
        result["status"] = "invalid_result"
        result["interpretation"] = "No conclusion: test returned a nonfinite result."
        return result
    if not 0 <= test.pvalue <= 1:
        result["status"] = "invalid_result"
        result["interpretation"] = "No conclusion: test returned a p-value outside [0, 1]."
        return result
    result["statistic"] = float(test.statistic)
    result["p_value"] = float(test.pvalue)
    result["status"] = "ok"
    result["interpretation"] = (
        "Evidence against the null at the unadjusted alpha; consider multiplicity."
        if test.pvalue <= alpha
        else "Insufficient evidence to reject the null; this does not establish equivalence."
    )
    return result


def adjust_pvalues_bh(pvalues: Iterable[float]) -> np.ndarray:
    """Return Benjamini-Hochberg adjusted p-values, preserving order and NaNs.

    NaNs denote unperformed/invalid tests and are not counted as performed
    tests. Report these exclusions and define the testing family in advance.
    Infinite values and finite p-values outside [0, 1] are rejected.
    """
    values = np.asarray(list(pvalues), dtype=float)
    if values.ndim != 1:
        raise DataValidationError("pvalues must be a one-dimensional sequence")
    if np.isinf(values).any() or ((values < 0) | (values > 1)).any():
        raise DataValidationError("pvalues must be in [0, 1] or NaN")
    adjusted = np.full(values.shape, np.nan, dtype=float)
    valid_indices = np.flatnonzero(np.isfinite(values))
    count = len(valid_indices)
    if not count:
        return adjusted
    ordered_indices = valid_indices[np.argsort(values[valid_indices], kind="stable")]
    ranked = values[ordered_indices] * count / np.arange(1, count + 1)
    corrected = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[ordered_indices] = np.clip(corrected, 0.0, 1.0)
    return adjusted
