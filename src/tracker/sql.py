from __future__ import annotations
from pathlib import Path
import duckdb

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parents[2]
SILVER = ROOT / "data" / "silver" / "constructor_race_points.parquet"
GOLD = ROOT / "data" / "gold" / "constructor_monthly.parquet"
GOLD.parent.mkdir(parents=True, exist_ok=True)


# ---------- Main ----------
def main() -> None:
    # Opens an in-memory DuckDB connection
    con = duckdb.connect(database=":memory:")

    # Makes sure the Parquet extension is available
    con.execute("INSTALL parquet; LOAD parquet;")

    # Creates a view over the silver Parquet file so we can query it like a table.
    con.execute(f"""
        CREATE OR REPLACE VIEW silver AS
        SELECT * FROM read_parquet('{SILVER.as_posix()}')
    """)

    # Build a monthly aggregate: one row per (constructor_id, m).
    # SUM(points) gives the total per month. We keep constructor_name for readability.
    # Then we use a window function LAG(...) to look at *previous* month’s total for the same constructor.
    # Finally, we compute MoM. If previous month is NULL or 0, we return NULL (not 0) to avoid fake growth.
    result = con.execute("""
        WITH monthly AS (
          SELECT
            constructor_id,
            constructor_name,
            m,
            SUM(points) AS points_m
          FROM silver
          GROUP BY 1,2,3
        ),
        with_prev AS (
          SELECT
            constructor_id,
            constructor_name,
            m,
            points_m,
            LAG(points_m) OVER (PARTITION BY constructor_id ORDER BY m) AS prev_points_m
          FROM monthly
        )
        SELECT
          constructor_id,
          constructor_name,
          m,
          points_m,
          CASE
            WHEN prev_points_m IS NULL OR prev_points_m = 0 THEN NULL
            ELSE (points_m - prev_points_m) * 1.0 / prev_points_m
          END AS mom_growth
        FROM with_prev
        ORDER BY constructor_id, m
    """).df()

    result.to_parquet(GOLD, index=False)

    print("Gold rows:", len(result))
    print("Unique keys:", result[["constructor_id", "m"]].drop_duplicates().shape[0])
    print("Months:", result["m"].min(), "→", result["m"].max(), "| n_unique:", result["m"].nunique())
    print("Saved ->", GOLD)


if __name__ == "__main__":
    main()
