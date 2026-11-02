"""Figure 1: exact match by template and reader."""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FW = ROOT / "results" / "r3" / "frameworks"
OUT = ROOT.parents[1] / "04-writing" / "figures" / "templates"

rows = []
for f in sorted(FW.glob("fw_hotpot_*.jsonl")):
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))

cell = defaultdict(list)
for r in rows:
    if r["condition"] == "rag":
        cell[(r["model"], r["template"])].append(r["em"])

readers = [("llama3.2:1b", "Llama-3.2-1B"), ("qwen2.5:3b", "Qwen2.5-3B"),
           ("qwen2.5:7b", "Qwen2.5-7B"), ("llama3:8b", "Llama3-8B")]
templates = [("FR", "FlashRAG"), ("LI", "LlamaIndex"), ("LC", "LangChain")]
colors = {"FR": "#55A868", "LI": "#4C72B0", "LC": "#C44E52"}

fig, ax = plt.subplots(figsize=(6.4, 3.6))
w = 0.26
x = np.arange(len(readers))
for i, (tk, tname) in enumerate(templates):
    vals = [100 * np.mean(cell.get((m, tk), [0])) if cell.get((m, tk)) else 0 for m, _ in readers]
    ax.bar(x + (i - 1) * w, vals, w, label=tname, color=colors[tk])
ax.set_xticks(x)
ax.set_xticklabels([n for _, n in readers], fontsize=9)
ax.set_ylabel("Exact match (HotpotQA, retrieval)")
ax.set_title("Same reader, same documents, three published default templates", fontsize=10)
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
p = OUT / "fig_templates.png"
fig.savefig(p, dpi=200)
print("wrote", p)
for m, name in readers:
    print(name, {tk: round(100 * np.mean(cell.get((m, tk), [0])), 1) for tk, _ in templates})
