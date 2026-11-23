import csv
from pathlib import Path

rows = []
for n in (1, 2, 3, 4):
    f = Path(f"results/locust_{n}_containers_stats.csv")
    if not f.exists():
        continue
    for r in csv.DictReader(f.open()):
        if r["Name"] == "/api/predict":
            rows.append((n, r))
lines = [
    "| Containers | Requests | Failures | Median (ms) | 95th pct (ms) | 99th pct (ms) | Avg (ms) | Req/s |",
    "|---|---|---|---|---|---|---|---|",
]
for n, r in rows:
    lines.append(
        f"| {n} | {r['Request Count']} | {r['Failure Count']} | {r['Median Response Time']} | {r['95%']} | {r['99%']} | "
        f"{float(r['Average Response Time']):.0f} | {float(r['Requests/s']):.1f} |"
    )
out = "\n".join(lines)
Path("results/summary.md").write_text(out + "\n")
print(out)
