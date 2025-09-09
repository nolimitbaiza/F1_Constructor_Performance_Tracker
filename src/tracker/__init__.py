# src/pipeline/ingest_f1.py
from __future__ import annotations
from pathlib import Path
import pandas as pd

NA = ["\\N"]  # treat this as missing during CSV load
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
BRONZE = ROOT / "data" / "bronze"

BRONZE.mkdir(parents=True, exist_ok=True)


# --- loaders (keep only columns we need) ---

# noinspection PyTypeChecker
def load_races(path: Path) -> pd.DataFrame:
    usecols = ["raceId", "date"]
    df = pd.read_csv(path, usecols=usecols, na_values=NA, parse_dates=["date"], keep_default_na=True)
    df = df.rename(columns={"raceId": "race_id", "date": "race_date"})
    return df


# noinspection PyTypeChecker
def load_constructors(path: Path) -> pd.DataFrame:
    usecols = ["constructorId", "name"]
    df = pd.read_csv(path, usecols=usecols, na_values=NA, keep_default_na=True)
    df = df.rename(columns={"constructorId": "constructor_id", "name": "constructor_name"})
    return df


# noinspection PyTypeChecker
def load_constructor_results(path: Path) -> pd.DataFrame:
    usecols = ["raceId", "constructorId", "points"]
    df = pd.read_csv(path, usecols=usecols, na_values=NA, keep_default_na=True)
    df = df.rename(columns={"raceId": "race_id", "constructorId": "constructor_id"})
    # ensure numeric points
    df["points"] = pd.to_numeric(df["points"], errors="coerce")
    return df


# --- build bronze table ---

def build_bronze(races: Path, cons: Path, cons_results: Path) -> pd.DataFrame:
    races = load_races(races)
    constructors = load_constructors(cons)
    cons_results = load_constructor_results(cons_results)

    # join results -> races to get the date
    tmp = cons_results.merge(races, on="race_id", how="left")
    # join -> constructors to get readable name
    bronze_data = tmp.merge(constructors, on="constructor_id", how="left")

    # sanity checks
    assert bronze_data[["race_id", "constructor_id"]].drop_duplicates().shape[0] == bronze_data.shape[0], \
        "(race_id, constructor_id) must be unique in bronze"
    assert pd.api.types.is_datetime64_any_dtype(bronze_data["race_date"]), "race_date must be datetime"

    # reorder columns
    bronze_data = bronze_data[["race_id", "race_date", "constructor_id", "constructor_name", "points"]]
    return bronze_data


def save_parquet(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)


if __name__ == "__main__":
    races_csv = RAW / "races.csv"
    cons_csv = RAW / "constructors.csv"
    cons_results_csv = RAW / "constructor_results.csv"

    bronze = build_bronze(races_csv, cons_csv, cons_results_csv)

    # quick null checks
    print("Rows:", len(bronze))
    print(bronze[["points", "race_date"]].isna().sum())

    out = BRONZE / "race_constructor_points.parquet"
    save_parquet(bronze, out)
    print("Saved ->", out)
