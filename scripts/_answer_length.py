"""Answer length by template: the mechanism behind the EM gap."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FW = ROOT / "results" / "r3" / "frameworks"

rows = []
for f in sorted(FW.glob("fw_hotpot_qwen253b_n500.jsonl")):
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))

by = defaultdict(list)
for r in rows:
    if r["condition"] == "rag":
        by[r["template"]].append(r)

print(f"{'template':<11}{'n':>5}{'mean chars':>12}{'median':>9}{'EM':>7}{'contains':>10}")
for t in ["FR", "LI", "LC", "LC_minus", "LC_neutral"]:
    rs = by.get(t, [])
    if not rs:
        continue
    L = [len(r["answer"]) for r in rs]
    em = 100 * np.mean([r["em"] for r in rs])
    ct = 100 * np.mean([r.get("contains", 0) for r in rs])
    print(f"{t:<11}{len(rs):>5}{np.mean(L):>12.0f}{np.median(L):>9.0f}{em:>7.1f}{ct:>10.1f}")
