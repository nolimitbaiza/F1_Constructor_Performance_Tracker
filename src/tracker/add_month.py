from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BRONZE = ROOT / "data" / "bronze"
INP = BRONZE / "race_constructor_points.parquet"
OUT_SIMPLE = BRONZE / "race_constructor_points_with_month.parquet"
OUT_PARTITIONED = BRONZE / "by_month"


def load_bronze(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    expected = {"race_id", "race_date", "constructor_id", "constructor_name", "points"}
    missing = expected - set(df.columns)
    assert not missing, f"Missing columns: {missing}"
    return df


def add_month_col(df: pd.DataFrame) -> pd.DataFrame:
    """

    :type df: pd.DataFrame
    """
    if not pd.api.types.is_datetime64_any_dtype(df["race_date"]):
        df["race_date"] = pd.to_datetime(df["race_date"], errors="raise")

    if getattr(df["race_date"].dt, "tz", None) is not None:
        df["race_date"] = df["race_date"].dt.tz_localize(None)

    m = df["race_date"].dt.to_period("M").dt.to_timestamp("MS")
    df["m"] = m

    assert pd.api.types.is_datetime64_any_dtype(df["m"]), "m must be datetime64[ns]"
    assert (df["m"].dt.day == 1).all(), "m must be the first day of the month"

    return df


def assert_invariants(df_before: pd.DataFrame, df_after: pd.DataFrame) -> None:
    # same number of rows
    assert len(df_after) == len(df_before), "Row count changed unexpectedly"
    # uniqueness preserved
    assert df_after.duplicated(["race_id", "constructor_id"]).sum() == 0, "Duplicates appeared"
    # dtype check for m
    assert pd.api.types.is_datetime64_any_dtype(df_after["m"]), "m must be datetime"
    # first-day check
    assert (df_after["m"].dt.day == 1).all(), "m must be first day of month"


def save_simple(df: pd.DataFrame) -> Path:
    OUT_SIMPLE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_SIMPLE, index=False)
    return OUT_SIMPLE


def save_partitioned(df: pd.DataFrame) -> Path:
    """
     Returns the base output directory path.
    """

    # --- Preconditions: we require an 'm' column that is a real datetime -----------
    # Guard that the column exists; fail early if upstream steps were skipped.
    assert "m" in df.columns, "Expected column 'm' missing; run add_month_col first"

    # Guard that 'm' is a datetime64 (not a string/Period); partition values come from it.
    assert pd.api.types.is_datetime64_any_dtype(df["m"]), "'m' must be datetime64[ns]"

    # Optional sanity: every 'm' should be the first of its month; catches bad derivations.
    assert (df["m"].dt.day == 1).all(), "'m' must be the first day of the month"

    # --- Prepare the output root directory ----------------------------------------
    # OUT_PARTITIONED is a module-level Path like data/bronze/by_month.
    base = OUT_PARTITIONED

    # Ensure the directory exists (mkdir -p behavior with parents=True).
    base.mkdir(parents=True, exist_ok=True)

    # --- Iterate months and write a Parquet per partition -------------------------
    # We keep a running total to cross-check that we didn't lose/duplicate rows.
    total_rows = 0

    # Sort by 'm' for predictable folder ordering, then group rows by month value.
    for month_ts, part_df in df.sort_values("m").groupby("m", sort=True):
        # `month_ts` is a pandas Timestamp; we'll turn it into a folder-friendly label.
        # We want EXACT format 'YYYY-MM-01' to match our medallion examples and make joins predictable.
        month_label = month_ts.strftime("%Y-%m-01")

        # Build the subfolder path like data/bronze/by_month/m=1950-05-01
        subdir = base / f"m={month_label}"

        # Create the partition directory if it doesn't exist yet.
        subdir.mkdir(parents=True, exist_ok=True)

        # Choose a simple, single-file name inside each partition.
        out_path = subdir / "part.parquet"

        # Write only the rows for this month partition (no index column).
        part_df.to_parquet(out_path, index=False)

        # Update our row counter to validate completeness.
        total_rows += len(part_df)

    # --- Postconditions: quick integrity check ------------------------------------
    # Confirm that the sum of all written rows equals the input row count.
    assert total_rows == len(df), f"Row mismatch: wrote {total_rows} but input had {len(df)}"

    # Return the base directory so callers can print/log where data landed.
    return base


def main(option: str = "A") -> None:
    before = load_bronze(INP)
    after = add_month_col(before.copy())
    assert_invariants(before, after)

    n_months = after["m"].nunique()
    print(f"rows={len(after)} months={n_months} range={after['m'].min()} → {after['m'].max()}")

    if option.upper() == "A":
        out = save_simple(after)
    else:
        out = save_partitioned(after)
    print("Saved ->", out)


if __name__ == "__main__":
    # Run A for now; switch to B to practice partitioning
    # main(option="A")
    bronze = load_bronze(INP)
    add_month_col(bronze.copy())
