"""Camera-ready verification for the template paper (v2, includes 8B, anchor, truncation).

Every number quoted in the paper is recomputed from the raw rows; the one external
reference (FlashRAG's reported F1) is checked for being quoted consistently.
"""
import importlib.util
import json
import math
import string
import sys
from collections import defaultdict
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


def refused(a):
    a = (a or "").strip().lower()
    return (a == "" or a.startswith("unknown") or "don't know" in a or "do not know" in a
            or "cannot be determined" in a or "not provided" in a)


def mcnemar(a, b):
    n01 = sum(1 for x, y in zip(a, b) if x and not y)
    n10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


rows = []
for f in sorted(FW.glob("fw_*.jsonl")):
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))

cell = defaultdict(list)
for r in rows:
    cell[(r["dataset"], r["model"], r["template"], r["condition"])].append(r)


def agg(ds, model, tmpl, cond, key):
    rs = cell.get((ds, model, tmpl, cond), [])
    if not rs:
        return None
    if key == "abstain":
        return 100 * np.mean([refused(r["answer"]) for r in rs])
    if key == "len":
        return float(np.mean([len(r["answer"]) for r in rs]))
    return 100 * np.mean([r.get(key, 0) for r in rs])


checks = []
# 3B headline
for ds, vals in [("hotpot", (46.4, 0.0, 0.0)), ("2wiki", (37.8, 0.0, 0.0)),
                 ("musique", (12.2, 0.0, 0.0))]:
    for tmpl, exp in zip(["FR", "LI", "LC"], vals):
        checks.append((f"{ds} {tmpl} rag EM", exp, round(agg(ds, "qwen2.5:3b", tmpl, "rag", "em"), 1)))
for ds, exp_fr, exp_lc in [("hotpot", 50.6, 52.2), ("2wiki", 44.4, 43.0), ("musique", 15.2, 10.2)]:
    checks.append((f"{ds} FR rag contains", exp_fr, round(agg(ds, "qwen2.5:3b", "FR", "rag", "contains"), 1)))
    checks.append((f"{ds} LC rag contains", exp_lc, round(agg(ds, "qwen2.5:3b", "LC", "rag", "contains"), 1)))
# clause effect 3B
for ds, exp in [("hotpot", 91.2), ("2wiki", 79.0), ("musique", 95.0)]:
    checks.append((f"{ds} LC closed abstain", exp, round(agg(ds, "qwen2.5:3b", "LC", "closed", "abstain"), 1)))
for ds, exp in [("hotpot", 1.0), ("2wiki", 0.0), ("musique", 2.0)]:
    checks.append((f"{ds} LC_minus closed abstain", exp, round(agg(ds, "qwen2.5:3b", "LC_minus", "closed", "abstain"), 1)))
# lengths
for tmpl, exp in [("FR", 19), ("LI", 134), ("LC", 133), ("LC_minus", 139), ("LC_neutral", 141)]:
    checks.append((f"hotpot {tmpl} mean length", exp, round(agg("hotpot", "qwen2.5:3b", tmpl, "rag", "len"), 0)))
# 8B
for name, exp, got in [
    ("8B FR rag EM", 54.0, agg("hotpot", "llama3:8b", "FR", "rag", "em")),
    ("8B LC rag EM", 2.0, agg("hotpot", "llama3:8b", "LC", "rag", "em")),
    ("8B LI rag EM", 0.0, agg("hotpot", "llama3:8b", "LI", "rag", "em")),
    ("8B LC rag abstain", 22.0, agg("hotpot", "llama3:8b", "LC", "rag", "abstain")),
    ("8B LC closed abstain", 80.0, agg("hotpot", "llama3:8b", "LC", "closed", "abstain")),
    ("8B FR closed F1 (anchor)", 24.9, agg("hotpot", "llama3:8b", "FR", "closed", "f1")),
    ("8B FR rag F1", 67.6, agg("hotpot", "llama3:8b", "FR", "rag", "f1")),
    ("7B FR rag EM", 51.5, agg("hotpot", "qwen2.5:7b", "FR", "rag", "em")),
    ("7B LC rag abstain", 0.0, agg("hotpot", "qwen2.5:7b", "LC", "rag", "abstain")),
    ("1B LC rag abstain", 91.5, agg("hotpot", "llama3.2:1b", "LC", "rag", "abstain")),
    ("1B LI rag abstain", 0.0, agg("hotpot", "llama3.2:1b", "LI", "rag", "abstain")),
]:
    checks.append((name, exp, round(got, 1)))

# truncation
spec = importlib.util.spec_from_file_location("r3", ROOT / "scripts" / "run_r3.py")
r3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r3)
qids = [str(q) for q in r3.load_splits()["hotpot"]["held_out_qids"]][:500]
golds = {str(e.get("qid")): list(e["answers"]) for e in r3.load_examples_by_qids("hotpot", qids)}
for tmpl, exp in [("FR", 41.8), ("LI", 1.4), ("LC", 1.2)]:
    rs = [r for r in cell[("hotpot", "qwen2.5:3b", tmpl, "rag")]]
    vals = [em(" ".join((r["answer"] or "").split()[:3]), golds.get(r["qid"], [])) for r in rs]
    checks.append((f"{tmpl} first-3-words EM", exp, round(100 * np.mean(vals), 1)))

print(f"{'check':<40}{'paper':>9}{'recomputed':>12}  status")
bad = 0
for name, exp, got in checks:
    ok = abs(float(exp) - float(got)) <= max(0.15, abs(float(exp)) * 0.02)
    bad += not ok
    print(f"{name:<40}{exp:>9}{got:>12}  {'ok' if ok else 'FAIL'}")
print(f"\n{bad} mismatches out of {len(checks)}")

