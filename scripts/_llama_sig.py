"""Significance of the 1B clause effect: LC vs LI, retrieval condition."""
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FW = ROOT / "results" / "r3" / "frameworks"


def refused(a):
    a = (a or "").strip().lower()
    return (a == "" or a.startswith("unknown") or "don't know" in a or "do not know" in a
            or "cannot be determined" in a or "not provided" in a)


rows = []
for f in sorted(FW.glob("fw_hotpot_llama321b*.jsonl")):
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))

by = defaultdict(dict)
for r in rows:
    by[(r["template"], r["condition"])][r["qid"]] = r

for cond in ["rag", "closed"]:
    a, b = by[("LC", cond)], by[("LI", cond)]
    q = sorted(set(a) & set(b))
    ra = [refused(a[x]["answer"]) for x in q]
    rb = [refused(b[x]["answer"]) for x in q]
    n01 = sum(1 for x, y in zip(ra, rb) if x and not y)
    n10 = sum(1 for x, y in zip(ra, rb) if y and not x)
    n = n01 + n10
    k = min(n01, n10)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    ca = 100 * np.mean([a[x].get("contains", 0) for x in q])
    cb = 100 * np.mean([b[x].get("contains", 0) for x in q])
    print(f"1B {cond:<7} LC abstain {100*np.mean(ra):5.1f}%  LI {100*np.mean(rb):5.1f}%  "
          f"p={p:.1e} ({n01}/{n10})  contains LC {ca:.1f} LI {cb:.1f}")
