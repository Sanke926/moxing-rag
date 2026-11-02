"""Is the EM collapse caused by the prose tail? Truncate answers and re-score.

If the correct answer sits at the start of a long response, cutting the response to its
first few words should recover EM. If it does not, the long answer genuinely does not
contain a short answer at its front.
"""
import json
import string
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FW = ROOT / "results" / "r3" / "frameworks"
NUMW = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
        "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
DROP = {"the", "a", "an", "of", "and", "in", "on", "at", "to", "for", "by", "with"}


def norm(s):
    s = (s or "").lower().strip().translate(str.maketrans("", "", string.punctuation))
    return " ".join(NUMW.get(t, t) for t in s.split() if t not in DROP)


def em(p, golds):
    p = norm(p)
    return int(bool(p) and any(p == norm(g) for g in golds))


import importlib.util
spec = importlib.util.spec_from_file_location("r3", ROOT / "scripts" / "run_r3.py")
r3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r3)
splits = r3.load_splits()
qids = [str(q) for q in splits["hotpot"]["held_out_qids"]][:500]
golds = {str(e.get("qid")): list(e["answers"]) for e in r3.load_examples_by_qids("hotpot", qids)}

rows = []
for f in sorted(FW.glob("fw_hotpot_qwen253b_n500.jsonl")):
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))

by = {}
for r in rows:
    if r["condition"] == "rag":
        by.setdefault(r["template"], []).append(r)

print(f"{'template':<11}{'n':>5}{'full':>8}{'w=3':>7}{'w=5':>7}{'w=8':>7}{'w=15':>7}")
for t in ["FR", "LI", "LC", "LC_minus"]:
    rs = by.get(t, [])
    if not rs:
        continue
    full = 100 * np.mean([r["em"] for r in rs])
    line = f"{t:<11}{len(rs):>5}{full:>8.1f}"
    for w in (3, 5, 8, 15):
        vals = []
        for r in rs:
            words = (r["answer"] or "").split()[:w]
            vals.append(em(" ".join(words), golds.get(r["qid"], [])))
        line += f"{100*np.mean(vals):>7.1f}"
    print(line)

