from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BRONZE = ROOT / "data" / "bronze"
INP = BRONZE / "race_constructor_points.parquet"
OUT = BRONZE / "race_constructor_points_with_month.parquet"
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

    m = df["race_date"].dt.to_period("M").dt.to_timestamp(how="start")
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


def save(df: pd.DataFrame) -> Path:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    return OUT


def main() -> None:
    before = load_bronze(INP)
    after = add_month_col(before.copy())
    assert_invariants(before, after)

    n_months = after["m"].nunique()
    print(f"rows={len(after)} months={n_months} range={after['m'].min()} → {after['m'].max()}")

    out = save(after)
    print("Saved ->", out)


if __name__ == "__main__":
    main()
