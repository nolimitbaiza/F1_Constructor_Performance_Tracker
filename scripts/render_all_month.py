import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "gold" / "constructor_monthly.parquet"
CHARTS = ROOT / "reports" / "charts"
REPORTS = ROOT / "reports"
CHARTS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

# Load months present in gold
g = pd.read_parquet(GOLD)
months = (
    g["m"].dt.strftime("%Y-%m")
    .drop_duplicates()
    .sort_values()
    .tolist()
)

# Render a Top-10 PNG for every available month using a report module
for ym in months:
    subprocess.run(
        [sys.executable, "-m", "src.tracker.report", "--month", ym, "--top", "10"],
        check=True
    )

# Build a tiny dropdown HTML that swaps the image by month
options_js = ",".join([f"'{m}'" for m in months])
html = f"""
<!doctype html>
<html>
  <head><meta charset="utf-8"><title>F1 Top-10 by Month</title></head>
  <body>
    <label for="m">Month:</label>
    <select id="m"></select>
    <br/><br/>
    <img id="chart" width="900"/>
    <script>
      const months=[{options_js}];
      const sel=document.getElementById('m');
      months.forEach(x=>{{const o=document.createElement('option');o.value=x;o.textContent=x;sel.appendChild(o);}});
      function set(x){{document.getElementById('chart').src='charts/top10_'+x+'.png'; sel.value=x;}}
      set(months[months.length-1]); // default to latest month
      sel.onchange=()=>set(sel.value);
    </script>
  </body>
</html>
"""
(REPORTS / "index.html").write_text(html, encoding="utf-8")

print("Rendered", len(months), "charts into", CHARTS)
print("Open:", REPORTS / "index.html")
