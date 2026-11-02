"""Paired statistics for the template experiment.

For each dataset: McNemar on (contains gold) between templates, and the abstention
difference for the closed-book condition.
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FW = ROOT / "results" / "r3" / "frameworks"


def load():
    rows = []
    for f in sorted(FW.glob("fw_*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def mcnemar(a, b):
    n01 = sum(1 for x, y in zip(a, b) if x and not y)
    n10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = n01 + n10
    if n == 0:
        return 1.0, n01, n10
    k = min(n01, n10)
    p = 2 * sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, p), n01, n10


def main():
    rows = load()
    by = defaultdict(dict)
    for r in rows:
        by[(r["dataset"], r["model"], r["condition"])][(r["template"], r["qid"])] = r

    for ds in ["hotpot", "2wiki", "musique"]:
        for model in sorted({m for (d, m, c) in by if d == ds}):
            for pair in [("FR", "LC"), ("LI", "LC"), ("LC", "LC_minus"), ("LC", "LC_neutral")]:
                for cond in ["rag", "closed"]:
                    a = {q: r for (t, q), r in by[(ds, model, cond)].items() if t == pair[0]}
                    b = {q: r for (t, q), r in by[(ds, model, cond)].items() if t == pair[1]}
                    q = sorted(set(a) & set(b))
                    if len(q) < 50:
                        continue
                    ca = [a[x].get("contains", 0) for x in q]
                    cb = [b[x].get("contains", 0) for x in q]
                    p, n01, n10 = mcnemar(ca, cb)
                    da = 100 * (np.mean(ca) - np.mean(cb))
                    aa = 100 * np.mean([1 if (a[x]["answer"] or "").strip() == "" else 0 for x in q])
                    print(f"{ds:<8}{model:<12}{pair[0]:<3} vs {pair[1]:<10}{cond:<7}"
                          f"n={len(q):<5} dContains={da:+6.1f}  p={p:.2e}  ({n01}/{n10})")


if __name__ == "__main__":
    main()
