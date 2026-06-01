from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SELECTED_ZONES: Tuple[str, ...] = (
    "NL",
    "GER",  # mapped to DE-LU.csv
    "DK1",
    "DK2",
    "SE1",
    "SE2",
    "SE3",
    "SE4",
    "NO1",
    "NO2",
    "NO3",
    "NO4",
    "NO5",
    "PL",
)


ZONE_FILENAME_OVERRIDES: Dict[str, str] = {
    # The dataset commonly uses DE-LU for Germany (DE+LU bidding zone)
    "GER": "DE-LU.csv",
}


def get_loads_dir(explicit_dir: Optional[Path] = None) -> Path:
    """Resolve the loads directory.

    Falls back to <repo_root>/data/loads relative to this file.
    """
    if explicit_dir is not None:
        return explicit_dir
    return (Path(__file__).resolve().parent / "data" / "loads").expanduser()


def list_existing_zone_files(loads_dir: Path) -> Dict[str, Path]:
    """Map each requested zone to an existing file path if present.

    Missing files are ignored.
    """
    mapping: Dict[str, Path] = {}
    for zone in SELECTED_ZONES:
        filename = ZONE_FILENAME_OVERRIDES.get(zone, f"{zone}.csv")
        candidate = loads_dir / filename
        if candidate.exists():
            mapping[zone] = candidate
    return mapping


def load_zone_dataframe(csv_path: Path, zone: str) -> pd.DataFrame:
    """Load a single zone CSV with columns: timestamp, Actual Load, Forecasted Load.

    Only the Actual Load is retained.
    """
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    # Normalize expected columns
    if "Actual Load" not in df.columns:
        raise ValueError(f"Missing 'Actual Load' in {csv_path}")
    if "timestamp" not in df.columns:
        raise ValueError(f"Missing 'timestamp' in {csv_path}")

    df = df[["timestamp", "Actual Load"]].rename(columns={"Actual Load": "actual_load"})
    df.insert(1, "zone", zone)
    # Drop rows with missing/invalid values
    df = df.dropna(subset=["timestamp", "actual_load"]).reset_index(drop=True)
    return df


def load_all_selected_zones(loads_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load and concatenate Actual Load time series for all selected zones.

    Uses only data within years 2021–2024 (inclusive).
    Returns a DataFrame with columns: [timestamp, zone, actual_load].
    Missing zones are silently skipped.
    """
    base_dir = get_loads_dir(loads_dir)
    zone_files = list_existing_zone_files(base_dir)

    if not zone_files:
        raise FileNotFoundError(
            f"No matching CSV files found in {base_dir}. Expected any of: "
            + ", ".join(
                ZONE_FILENAME_OVERRIDES.get(z, f"{z}.csv") for z in SELECTED_ZONES
            )
        )

    frames: List[pd.DataFrame] = []
    for zone, path in zone_files.items():
        try:
            frames.append(load_zone_dataframe(path, zone))
        except Exception as exc:
            # Skip unreadable/malformed files while proceeding with others
            print(f"Warning: skipping {path} due to error: {exc}")

    if not frames:
        raise RuntimeError("No data could be loaded for the selected zones.")

    all_df = pd.concat(frames, axis=0, ignore_index=True)
    # Ensure proper dtypes
    all_df["actual_load"] = pd.to_numeric(all_df["actual_load"], errors="coerce")
    all_df = all_df.dropna(subset=["actual_load"]).reset_index(drop=True)

    # Filter to requested period: 2021-01-01 through 2024-12-31 inclusive
    year = all_df["timestamp"].dt.year
    all_df = all_df[(year >= 2021) & (year <= 2024)].reset_index(drop=True)
    return all_df


def compute_hourly_usage_weights(loads_df: pd.DataFrame) -> List[float]:
    """Compute 24 hourly weights using historical Actual Load as a proxy for usage.

    Method: form a histogram over hour-of-day for all timestamps, using the Actual
    Load as per-sample weight. The resulting normalized bin heights sum to 1 and
    represent the probability of usage by hour.
    """
    if loads_df.empty:
        return [0.0] * 24

    hours = loads_df["timestamp"].dt.hour.to_numpy()
    weights = loads_df["actual_load"].to_numpy()

    # Keep finite, positive loads only (non-positive loads are not meaningful usage)
    finite_mask = np.isfinite(hours) & np.isfinite(weights) & (weights > 0)
    hours = hours[finite_mask]
    weights = weights[finite_mask]

    if hours.size == 0:
        return [0.0] * 24

    # 24 bins: [0,1), [1,2), ..., [23,24)
    bin_edges = np.arange(25)
    load_by_hour, _ = np.histogram(hours, bins=bin_edges, weights=weights)
    total = load_by_hour.sum()
    if total <= 0:
        return [0.0] * 24

    hourly_weights = (load_by_hour / total).astype(float)
    return hourly_weights.tolist()


def plot_load_value_distribution(
    loads_df: pd.DataFrame, *, save_path: Optional[Path] = None
) -> Path:
    """Plot the overall distribution of Actual Load values across all selected zones."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(loads_df["actual_load"], bins=60, color="#7e57c2", alpha=0.85)
    ax.set_title("Distribution of Actual Loads (Selected Zones, 2021–2024)")
    ax.set_xlabel("Actual Load")
    ax.set_ylabel("Frequency")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path is None:
        save_path = Path("loads_distribution.png")
    save_path = save_path.resolve()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_hourly_weight_bars(
    hourly_weights: Iterable[float], *, save_path: Optional[Path] = None
) -> Path:
    """Plot a bar chart of the 24 hourly usage weights."""
    weights = np.asarray(list(hourly_weights), dtype=float)
    if weights.shape[0] != 24:
        raise ValueError("hourly_weights must contain exactly 24 values")

    hours = np.arange(24)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(hours, weights, color="#1f77b4", edgecolor="#0f3d63")
    ax.set_title("Hourly Usage Weights inferred from Loads (2021–2024)")
    ax.set_xlabel("Hour of Day (0-23)")
    ax.set_ylabel("Weight (Probability)")
    ax.set_xticks(hours)
    ax.set_ylim(0, max(0.001, weights.max() * 1.15))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path is None:
        save_path = Path("hourly_load_weights.png")
    save_path = save_path.resolve()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def main(loads_dir: Optional[str] = None) -> List[float]:
    base_dir = Path(loads_dir) if loads_dir is not None else None
    loads_df = load_all_selected_zones(base_dir)

    # Create plots
    dist_path = plot_load_value_distribution(loads_df)
    weights = compute_hourly_usage_weights(loads_df)
    weights_path = plot_hourly_weight_bars(weights)

    print(f"Saved distribution plot to: {dist_path}")
    print(f"Saved hourly weights plot to: {weights_path}")
    print("24-hour weights (sum=1.0):")
    print([round(w, 6) for w in weights])

    return weights


if __name__ == "__main__":
    # When run directly, use the default relative data directory
    main()
