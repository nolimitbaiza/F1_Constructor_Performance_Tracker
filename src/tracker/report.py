from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parents[2]
GOLD = ROOT / "data" / "gold" / "constructor_monthly.parquet"
OUTDIR = ROOT / "reports" / "charts"
OUTDIR.mkdir(parents=True, exist_ok=True)


def load_gold() -> pd.DataFrame:
    """Loads gold and normalizes types."""
    df = pd.read_parquet(GOLD)
    df["m"] = pd.to_datetime(df["m"])
    return df


def plot_topk_for_month(df: pd.DataFrame, ym: str, k: int = 10) -> Path:
    """
    Creates a horizontal bar chart of top-K constructors by points for a given month.
    ym: 'YYYY-MM' (e.g., '2012-08' or '1993-04').
    Returns the path to the saved PNG.
    """
    # Converts 'YYYY-MM' to the canonical first-of-month timestamp used in gold.
    month = pd.to_datetime(ym + "-01")

    # Filters to the selected month; picks the top K by points_m.
    sub = (df[df["m"] == month].nlargest(k, "points_m")[["constructor_name", "points_m"]])

    # If the month has no rows (e.g., no races), gives a helpful error early.
    if sub.empty:
        raise SystemExit(f"No data for month {ym}. Try another month shown in gold.")

    # Builds the chart; sorts ascending so the biggest bar appears at the top.
    fig, ax = plt.subplots(figsize=(8, 5))
    sub.sort_values("points_m").plot(
        kind="barh", x="constructor_name", y="points_m",
        ax=ax, legend=False
    )
    ax.set_title(f"Top {k} Constructors — {ym}")
    ax.set_xlabel("Points in Month")

    # Saves chart into reports/charts with a consistent name.
    out = OUTDIR / f"top10_{ym}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main() -> None:
    # Parses CLI flags: --month 2012-08 --top 10
    p = argparse.ArgumentParser()
    p.add_argument("--month", required=True, help="YYYY-MM (e.g., 2012-08)")
    p.add_argument("--top", type=int, default=10)
    args = p.parse_args()

    # Loads gold once and generate the requested chart.
    gold = load_gold()
    out = plot_topk_for_month(gold, args.month, args.top)
    print("Saved ->", out)


if __name__ == "__main__":
    main()
