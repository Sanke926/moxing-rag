"""Analysis for the published-template experiment.

Primary (within LangChain, byte-identical except the sentence):
    LC vs LC_minus vs LC_neutral vs LC_cleared
Secondary (verbatim defaults of three frameworks):
    LC vs LI vs FR
Both under two conditions: rag (10 docs) and closed (no context).

Reports EM, F1, contains-gold, and the abstention rate, plus the closed-vs-rag gap
under each template -- the comparison that flips.
"""
import json
import string
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FW = ROOT / "results" / "r3" / "frameworks"
NUMW = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
        "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}


def refused(a):
    a = (a or "").strip().lower()
    return (a.startswith("unknown") or "don't know" in a or "do not know" in a
            or a.startswith("i don't know") or "cannot be determined" in a
            or "not provided" in a or a == "")


def load():
    rows = []
    for f in sorted(FW.glob("fw_*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    rows = load()
    by = defaultdict(list)
    for r in rows:
        by[(r["dataset"], r["model"], r["template"], r["condition"])].append(r)

    print(f"{'dataset':<8}{'model':<12}{'template':<11}{'cond':<7}{'n':>5}"
          f"{'EM':>7}{'F1':>7}{'contains':>10}{'abstain':>9}")
    for (ds, model, tk, cond), rs in sorted(by.items()):
        n = len(rs)
        em = 100 * np.mean([r["em"] for r in rs])
        f1 = 100 * np.mean([r.get("f1", 0) for r in rs])
        ct = 100 * np.mean([r.get("contains", 0) for r in rs])
        ab = 100 * np.mean([refused(r["answer"]) for r in rs])
        print(f"{ds:<8}{model:<12}{tk:<11}{cond:<7}{n:>5}{em:>7.1f}{f1:>7.1f}"
              f"{ct:>10.1f}{ab:>9.1f}")

    print("\nclosed minus rag, per template (positive = closed book looks better):")
    for (ds, model, tk), _ in sorted({(d, m, t): 1 for (d, m, t, c) in by}.items()):
        a = by.get((ds, model, tk, "closed"), [])
        b = by.get((ds, model, tk, "rag"), [])
        if not a or not b:
            continue
        da = 100 * (np.mean([r["em"] for r in a]) - np.mean([r["em"] for r in b]))
        dc = 100 * (np.mean([r.get("contains", 0) for r in a])
                    - np.mean([r.get("contains", 0) for r in b]))
        print(f"  {ds:<8}{model:<12}{tk:<11} dEM={da:+6.1f}  dContains={dc:+6.1f}")


if __name__ == "__main__":
    main()
