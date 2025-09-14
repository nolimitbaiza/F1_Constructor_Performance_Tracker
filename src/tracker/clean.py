from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BRONZE = ROOT / "data" / "bronze"
SILVER = ROOT / "data" / "silver"
SILVER.mkdir(parents=True, exist_ok=True)

INP_WITH_MONTH = BRONZE / "race_constructor_points_with_month.parquet"
INP_NO_MONTH = BRONZE / "race_constructor_points.parquet"
OUT_SILVER = SILVER / "constructor_race_points.parquet"


def _expect_columns(df: pd.DataFrame, cols: set[str]) -> None:
    """Fails fast if expected columns are missing."""
    missing = cols - set(df.columns)
    assert not missing, f"Missing columns: {missing}"


def _ensure_datetime(series: pd.Series) -> pd.Series:
    """Guarantees datetime64[ns] and strip timezone if present."""
    if not pd.api.types.is_datetime64_any_dtype(series):
        series = pd.to_datetime(series, errors="raise")
    if getattr(series.dt, "tz", None) is not None:
        series = series.dt.tz_localize(None)
    return series


def _ensure_month(df: pd.DataFrame) -> pd.DataFrame:
    """Adds/normalizes the month column 'm' (first day of month, datetime64[ns])."""
    if "m" in df.columns:
        df["m"] = pd.to_datetime(df["m"], errors="raise")
    else:
        df["m"] = df["race_date"].dt.to_period("M").dt.to_timestamp(how="start")
    assert pd.api.types.is_datetime64_any_dtype(df["m"]), "m must be datetime64[ns]"
    assert (df["m"].dt.day == 1).all(), "m must be the first day of the month"
    return df


def _dedupe_by_pair(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Removes duplicates on (race_id, constructor_id).
    Keeps the row with the highest 'points' (NaN sorts last, so a valid number wins).
    Returns the cleaned df and number of rows dropped.
    """
    before = len(df)
    df = df.sort_values(["race_id", "constructor_id", "points"], ascending=[True, True, False])
    df = df.drop_duplicates(subset=["race_id", "constructor_id"], keep="first")
    dropped = before - len(df)
    assert df.duplicated(["race_id", "constructor_id"]).sum() == 0, "Duplicates remain after dedupe"
    return df, dropped


def _clean_values(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Enforces numeric types and clamp impossible values.
    Returns df and an 'issues' dict for logging.
    """
    issues: dict[str, int] = {}

    df["points"] = pd.to_numeric(df["points"], errors="coerce")

    neg_mask = df["points"] < 0
    issues["neg_points_found"] = int(neg_mask.sum())
    if issues["neg_points_found"] > 0:
        df.loc[neg_mask, "points"] = pd.NA

    issues["missing_points"] = int(df["points"].isna().sum())
    issues["missing_constructor_name"] = int(df["constructor_name"].isna().sum())

    return df, issues


# ---------- Main pipeline function ----------
def main() -> None:
    """Builds the SILVER table from BRONZE with clear quality logs."""
    inp = INP_WITH_MONTH if INP_WITH_MONTH.exists() else INP_NO_MONTH
    df = pd.read_parquet(inp)

    _expect_columns(df, {"race_id", "race_date", "constructor_id", "constructor_name", "points"})
    df["race_date"] = _ensure_datetime(df["race_date"])

    df = _ensure_month(df)

    df, dropped = _dedupe_by_pair(df)

    df, issues = _clean_values(df)

    df = df[["race_id", "race_date", "m", "constructor_id", "constructor_name", "points"]]

    OUT_SILVER.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_SILVER, index=False)

    print(f"Loaded from: {inp}")
    print(f"Rows after dedupe: {len(df)}  (dropped={dropped})")
    print(f"Months: {df['m'].nunique()}  range: {df['m'].min()} → {df['m'].max()}")
    for k, v in issues.items():
        print(f"{k}: {v}")
    print("Saved ->", OUT_SILVER)


if __name__ == "__main__":
    main()
