"""Figures replacing the span panel: EM vs containment, and the clause controls."""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parents[1] / "04-writing" / "figures" / "audit"


def refused(a):
    a = (a or "").strip().lower()
    return (a == "" or a.startswith("unknown") or "don't know" in a or "do not know" in a
            or "cannot be determined" in a or "not provided" in a)


rows = []
for f in sorted((ROOT / "results" / "r3" / "frameworks").glob("fw_hotpot_qwen253b_n500.jsonl")):
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
cell = defaultdict(list)
for r in rows:
    cell[(r["template"], r["condition"])].append(r)

tmpl = ["FR", "LI", "LC"]
names = ["FlashRAG", "LlamaIndex", "LangChain"]
em = [100 * np.mean([r["em"] for r in cell[(t, "rag")]]) for t in tmpl]
ct = [100 * np.mean([r["contains"] for r in cell[(t, "rag")]]) for t in tmpl]

fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3), sharey=True)
ax = axes[0]
b = ax.bar(names, em, color=["#55A868", "#4C72B0", "#C44E52"], width=0.55)
for bb, v in zip(b, em):
    ax.text(bb.get_x() + bb.get_width() / 2, v + 1.2, f"{v:.1f}", ha="center", fontsize=10, weight="bold")
ax.set_title("Exact match: 46-point spread", fontsize=10)
ax.set_ylabel("Score (%)")
ax.set_ylim(0, 62)

ax = axes[1]
b = ax.bar(names, ct, color=["#55A868", "#4C72B0", "#C44E52"], width=0.55)
for bb, v in zip(b, ct):
    ax.text(bb.get_x() + bb.get_width() / 2, v + 1.2, f"{v:.1f}", ha="center", fontsize=10, weight="bold")
ax.set_title("Containment: the same knowledge", fontsize=10)
ax.set_ylim(0, 62)
for ax in axes:
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "fig_flat.png", dpi=220)
print("wrote fig_flat")

# controls
vars_ = ["LC", "LC_minus", "LC_neutral", "LC_cleared"]
labels = ["LC\n(verbatim)", "LC_minus\n(deleted)", "LC_neutral\n(replaced)", "LC_cleared\n(empty)"]
vals = [100 * np.mean([refused(r["answer"]) for r in cell[(v, "closed")]]) for v in vars_]
fig, ax = plt.subplots(figsize=(5.4, 3.3))
b = ax.bar(labels, vals, color=["#C44E52", "#55A868", "#4C72B0", "#999999"], width=0.6)
for bb, v in zip(b, vals):
    ax.text(bb.get_x() + bb.get_width() / 2, v + 2, f"{v:.1f}", ha="center", fontsize=10, weight="bold")
ax.set_ylim(0, 108)
ax.set_ylabel("Abstention, empty context (%)")
ax.set_title("Length is not the cause: three controls", fontsize=10)
ax.grid(axis="y", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "fig_controls.png", dpi=220)
print("wrote fig_controls")
