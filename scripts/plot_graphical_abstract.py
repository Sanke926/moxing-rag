"""Figure 1: graphical abstract showing both effects in one panel."""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
FW = ROOT / "results" / "r3" / "frameworks"
OUT = ROOT.parents[1] / "04-writing" / "figures" / "templates"

rows = []
for f in sorted(FW.glob("fw_hotpot_qwen253b_n500.jsonl")):
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
cell = defaultdict(list)
for r in rows:
    cell[(r["template"], r["condition"])].append(r)


def refused(a):
    a = (a or "").strip().lower()
    return (a == "" or a.startswith("unknown") or "don't know" in a or "do not know" in a
            or "cannot be determined" in a or "not provided" in a)


def m(t, c, k):
    rs = cell.get((t, c), [])
    if not rs:
        return 0.0
    if k == "abstain":
        return 100 * sum(refused(r["answer"]) for r in rs) / len(rs)
    return 100 * sum(r.get(k, 0) for r in rs) / len(rs)


fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))

# --- left: template -> EM
ax = axes[0]
names = ["FlashRAG", "LlamaIndex", "LangChain"]
ems = [m("FR", "rag", "em"), m("LI", "rag", "em"), m("LC", "rag", "em")]
cols = ["#55A868", "#4C72B0", "#C44E52"]
bars = ax.bar(names, ems, color=cols, width=0.6)
for b, v in zip(bars, ems):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}", ha="center", fontsize=11, weight="bold")
ax.set_ylim(0, 56)
ax.set_ylabel("Exact match (HotpotQA)")
ax.set_title("Same reader, same documents, same questions", fontsize=10)
ax.grid(axis="y", alpha=0.3)
ax.annotate("", xy=(2, 4), xytext=(0, 44),
            arrowprops=dict(arrowstyle="<->", color="#444", lw=1.4, ls="--"))
ax.text(1.0, 30, "46 points\nfrom the template", ha="center", fontsize=9, color="#444",
        bbox=dict(fc="white", ec="none", alpha=0.85))

# --- right: clause on/off -> abstention
ax = axes[1]
labels = ["with the\nsentence", "sentence\ndeleted"]
vals = [m("LC", "closed", "abstain"), m("LC_minus", "closed", "abstain")]
bars = ax.bar(labels, vals, color=["#C44E52", "#55A868"], width=0.55)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.1f}%", ha="center", fontsize=11, weight="bold")
ax.set_ylim(0, 105)
ax.set_ylabel("Abstention, empty context (%)")
ax.set_title("One sentence decides the closed-book baseline", fontsize=10)
ax.grid(axis="y", alpha=0.3)

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle("The reported number is a property of the template", fontsize=12, weight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.94))
OUT.mkdir(parents=True, exist_ok=True)
p = OUT / "fig_graphical_abstract.png"
fig.savefig(p, dpi=220)
print("wrote", p)

