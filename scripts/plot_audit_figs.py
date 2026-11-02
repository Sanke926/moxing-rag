"""Figures 2-4 for the integrated audit paper."""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parents[1] / "04-writing" / "figures" / "audit"
OUT.mkdir(parents=True, exist_ok=True)


def refused(a):
    a = (a or "").strip().lower()
    return (a == "" or a.startswith("unknown") or "don't know" in a or "do not know" in a
            or "cannot be determined" in a or "not provided" in a)


# ---- Fig 2: abstention, three datasets, with vs without the clause ----
rows = []
for f in sorted((ROOT / "results" / "r3" / "frameworks").glob("fw_*_qwen253b_n500.jsonl")):
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
cell = defaultdict(list)
for r in rows:
    cell[(r["dataset"], r["template"], r["condition"])].append(r)

dsets = ["hotpot", "2wiki", "musique"]
labels = ["HotpotQA", "2Wiki", "MuSiQue"]
with_c = [100 * np.mean([refused(r["answer"]) for r in cell[(d, "LC", "closed")]]) for d in dsets]
without = [100 * np.mean([refused(r["answer"]) for r in cell[(d, "LC_minus", "closed")]]) for d in dsets]

fig, ax = plt.subplots(figsize=(5.2, 3.4))
x = np.arange(len(dsets))
w = 0.36
ax.bar(x - w / 2, with_c, w, label="with the sentence", color="#C44E52")
ax.bar(x + w / 2, without, w, label="sentence deleted", color="#55A868")
for i, (a, b) in enumerate(zip(with_c, without)):
    ax.text(i - w / 2, a + 2, f"{a:.1f}", ha="center", fontsize=9)
    ax.text(i + w / 2, b + 2, f"{b:.1f}", ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Abstention, empty context (%)")
ax.set_ylim(0, 108)
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_title("One sentence decides the closed-book baseline", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "fig_abstention.png", dpi=220)
print("wrote fig_abstention")

# ---- Fig 3: retrieval-pool degeneracy ----
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3))
ax = axes[0]
labels = ["supplied pool\n(10 candidates)", "expanded corpus\n(4,930 paragraphs)"]
vals = [97.4, 59.0]
bars = ax.bar(labels, vals, color=["#C44E52", "#4C72B0"], width=0.55)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}%", ha="center", fontsize=10, weight="bold")
ax.set_ylim(0, 116)
ax.set_ylabel("Gold paragraph retrieved (%)")
ax.set_title("Coverage depends on the pool", fontsize=10)

ax = axes[1]
bars = ax.bar(["BM25 top-10", "CE reranked top-10"], [33.2, 38.4],
              color=["#999999", "#55A868"], width=0.5)
for b, v in zip(bars, [33.2, 38.4]):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f}", ha="center", fontsize=10, weight="bold")
ax.set_ylim(0, 46)
ax.set_ylabel("Exact match")
ax.set_title("Reranking gain on the expanded corpus", fontsize=10)
for ax in axes:
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "fig_retrieval.png", dpi=220)
print("wrote fig_retrieval")

# ---- Fig 4: metric convention span ----
fig, ax = plt.subplots(figsize=(5.2, 3.4))
names = ["MuSiQue\nvote", "2Wiki\nvote", "Hotpot\nvote", "Hotpot\nSC@3"]
spans = [0.1092, 0.1311, 0.1084, 0.1885]
bars = ax.bar(names, spans, color=["#4C72B0", "#4C72B0", "#4C72B0", "#C44E52"], width=0.55)
for b, v in zip(bars, spans):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", fontsize=9)
ax.set_ylabel("AURC difference between conventions")
ax.set_ylim(0, 0.22)
ax.grid(axis="y", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_title("A confidence with few levels has an unstable score", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "fig_span.png", dpi=220)
print("wrote fig_span")


