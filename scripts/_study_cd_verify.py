"""Verification for Study C (pool) and Study D (metric), from stored artefacts."""
import json
import math
import random
import string
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results" / "r3"
NUMW = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
        "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
DROP = {"the", "a", "an", "of", "and", "in", "on", "at", "to", "for", "by", "with"}


def norm(s):
    s = (s or "").lower().strip().translate(str.maketrans("", "", string.punctuation))
    return " ".join(NUMW.get(t, t) for t in s.split() if t not in DROP)


def refused(p):
    p = (p or "").strip().lower()
    return p.startswith("unknown") or p in {"", "n/a", "none"}


def mcnemar(a, b):
    n01 = sum(1 for x, y in zip(a, b) if x and not y)
    n10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = n01 + n10
    if n == 0:
        return 1.0, n01, n10
    k = min(n01, n10)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n), n01, n10


checks = []

# ---------- C1: end to end ----------
rows = [json.loads(l) for l in (R / "bigcorpus" / "e2e_hotpot_500.jsonl")
        .read_text(encoding="utf-8").splitlines() if l.strip()]
bm = [r["arms"]["bm25"]["em"] for r in rows]
ce = [r["arms"]["ce"]["em"] for r in rows]
p, bm_only, ce_only = mcnemar(bm, ce)
checks += [("C1 BM25 top-10 EM", 33.2, round(100 * np.mean(bm), 1)),
           ("C1 CE reranked EM", 38.4, round(100 * np.mean(ce), 1)),
           ("C1 CE wins (discordant)", 43, ce_only),
           ("C1 BM25 wins (discordant)", 17, bm_only),
           ("C1 McNemar p (x10000)", 11, round(p * 10000))]

# ---------- C2: read-all vs oracle ----------
S = {}
for f in (R / "subset").glob("subset_hotpot_*.jsonl"):
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            S[r["qid"]] = r
E = {}
for f in (R / "promptab").glob("promptab_hotpot_E_qwen253b_n500_rows.jsonl"):
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            E[r["qid"]] = r
q = sorted(set(S) & set(E))
read_all = 100 * np.mean([int(E[x]["em"] > 0) for x in q])
oracle = 100 * np.mean([int(any(S[x]["runs"][n]["em"] for n in ("bm25", "dense", "ce"))) for x in q])
checks += [("C2 read-all EM", 46.5, round(read_all, 1)),
           ("C2 oracle-2 EM", 48.5, round(oracle, 1)),
           ("C2 headroom", 2.0, round(oracle - read_all, 1))]

# ---------- C3: gold present in the supplied pool ----------
import importlib.util
spec = importlib.util.spec_from_file_location("r3mod", ROOT / "scripts" / "run_r3.py")
r3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r3)
qids = [str(x) for x in r3.load_splits()["hotpot"]["held_out_qids"]][:500]
ex = r3.load_examples_by_qids("hotpot", qids)
present = sum(1 for e in ex
              if any(a.lower() in " ".join(e["docs"]).lower() for a in e["answers"]))
checks.append(("C3 gold in pool (%)", 97.4, round(100 * present / len(ex), 1)))

# ---------- D: span law ----------
conv = json.loads((R / "ties" / "convention_span.json").read_text(encoding="utf-8"))
by = {(c["dataset"], c["K"]): c for c in conv}
checks += [("D hotpot K=3 span", 0.1084, round(by[("hotpot", 3)]["span_conv"], 4)),
           ("D 2wiki K=3 span", 0.1311, round(by[("2wiki", 3)]["span_conv"], 4)),
           ("D musique K=3 span", 0.1092, round(by[("musique", 3)]["span_conv"], 4)),
           ("D hotpot K=5 span", 0.0730, round(by[("hotpot", 5)]["span_conv"], 4)),
           ("D hotpot K=7 span", 0.0565, round(by[("hotpot", 7)]["span_conv"], 4)),
           ("D largest tie group (%)", 46.2, round(100 * by[("hotpot", 3)]["max_group_share"], 1))]
# identity: span == group - trapz, on every row
worst = max(abs(c["span_conv"] - (c["group"] - c["trapz"])) for c in conv)
checks.append(("D span == group - trapz (max err)", 0.0, round(worst, 9)))
# monotone in the number of levels
for ds in ("hotpot", "2wiki"):
    mono = all(by[(ds, k)]["span_conv"] > by[(ds, k2)]["span_conv"]
               for k, k2 in ((3, 5), (5, 7)))
    checks.append((f"D {ds} span decreases in K", 1, int(mono)))

print(f"{'check':<36}{'paper':>10}{'recomputed':>12}  status")
bad = 0
for name, exp, got in checks:
    ok = abs(float(exp) - float(got)) <= max(0.02, abs(float(exp)) * 0.03)
    bad += not ok
    print(f"{name:<36}{exp:>10}{got:>12}  {'ok' if ok else 'FAIL'}")
print(f"\n{bad} mismatches out of {len(checks)}")

