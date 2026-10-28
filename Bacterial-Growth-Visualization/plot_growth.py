"""Plot supplied OD600 observations, or an explicitly synthetic CSV example."""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd


def load_growth(path):
    data = pd.read_csv(path)
    required = {"Time_Hours", "Strain", "OD600"}
    if not required <= set(data.columns) or data.empty:
        raise ValueError("CSV needs rows and columns Time_Hours, Strain, OD600")
    for column in ("Time_Hours", "OD600"):
        data[column] = pd.to_numeric(data[column], errors="raise")
        if not np.isfinite(data[column].to_numpy()).all():
            raise ValueError(f"{column} must contain finite numbers")
    if (data["Time_Hours"] < 0).any():
        raise ValueError("Time_Hours cannot be negative")
    if data["Strain"].isna().any() or data["Strain"].astype(str).str.strip().eq("").any():
        raise ValueError("Every observation needs a nonempty Strain")
    keys = ["Strain"]
    if "Replicate" in data:
        if data["Replicate"].isna().any() or data["Replicate"].astype(str).str.strip().eq("").any():
            raise ValueError("Replicate identifiers cannot be empty")
        keys.append("Replicate")
    if data.duplicated(keys + ["Time_Hours"]).any():
        raise ValueError("Duplicate strain/replicate/time observations; identify replicates explicitly")
    return data.sort_values(keys + ["Time_Hours"]), keys


def plot_growth(path, output, synthetic=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data, keys = load_growth(path)
    if "Data_Source" in data and data["Data_Source"].astype(str).str.contains("synthetic", case=False).any():
        synthetic = True
    fig, ax = plt.subplots(figsize=(8, 5))
    for group, rows in data.groupby(keys, sort=True):
        labels = group if isinstance(group, tuple) else (group,)
        label = " / ".join(str(x) for x in labels)
        ax.plot(rows["Time_Hours"], rows["OD600"], marker="o", label=label)
    ax.set_title("SYNTHETIC EXAMPLE — OD600 over time" if synthetic else "OD600 observations over time")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Optical density at 600 nm")
    ax.grid(alpha=0.3)
    ax.legend(title="Strain / replicate" if len(keys) == 2 else "Strain")
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Your observed growth CSV")
    source.add_argument("--example", action="store_true", help="Use the bundled synthetic demonstration")
    parser.add_argument("--output", type=Path, default=Path("outputs/bacterial_growth_plot.png"))
    args = parser.parse_args(argv)
    path = Path(__file__).with_name("example_growth.csv") if args.example else args.input
    try:
        output = plot_growth(path, args.output, synthetic=args.example)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    print(f"Plot saved to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
