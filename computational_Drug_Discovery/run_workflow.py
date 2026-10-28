"""Run the same steps as the five notebooks without requiring Jupyter."""
import argparse
from qsar_workflow import (
    acquire_data, exploratory_analysis, prepare_descriptors,
    random_forest_baseline, compare_regressors,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Archived educational AChE workflow")
    parser.add_argument("--source-csv", help="Optional local cache of the exact pinned archive")
    parser.add_argument("--padel-csv", help="Optional PaDEL output with record IDs in Name")
    parser.add_argument("--skip-model-comparison", action="store_true")
    args = parser.parse_args()
    records = acquire_data(args.source_csv)
    exploratory_analysis(records)
    table, features = prepare_descriptors(records, args.padel_csv)
    random_forest_baseline(table, features)
    if not args.skip_model_comparison:
        compare_regressors(table, features)
