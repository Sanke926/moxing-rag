"""Figure 1 in a conventional journal style: serif type, thin rules, real axes, no color blocks."""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parents[1] / "04-writing" / "figures" / "audit" / "fig_overview.png"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "dejavuserif",
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "font.size": 8,
})

INK = "#1a1a1a"
ACC = "#8c2d2d"          # single accent, used sparingly
GREY = "#6e6e6e"

fig = plt.figure(figsize=(7.1, 3.5))
gs = fig.add_gridspec(1, 4, left=0.075, right=0.985, top=0.86, bottom=0.30, wspace=0.62)

axs = [fig.add_subplot(gs[0, i]) for i in range(4)]

# ---------- (a) template ----------
ax = axs[0]
labels = ["FR", "LI", "LC"]
vals = [46.4, 0.0, 0.0]
x = np.arange(3)
ax.bar(x, vals, width=0.55, color=["white", "white", "white"], edgecolor=INK, linewidth=0.8)
ax.bar([0], [46.4], width=0.55, color="#d9d9d9", edgecolor=INK, linewidth=0.8)
for xi, v in zip(x, vals):
    ax.text(xi, v + 1.6, f"{v:.1f}", ha="center", va="bottom", fontsize=6.6)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=7)
ax.set_ylim(0, 60)
ax.set_ylabel("exact match", fontsize=7)
ax.set_title("(a) template", fontsize=7.8, pad=5, loc="left", style="italic")
ax.annotate("", xy=(0.48, 52.5), xytext=(1.62, 52.5),
            arrowprops=dict(arrowstyle="<->", lw=0.7, color=ACC))
ax.text(1.05, 54.2, "46", ha="center", fontsize=7.2, color=ACC, weight="bold")
ax.spines[["top", "right"]].set_visible(False)

# ---------- (b) instruction ----------
ax = axs[1]
x = np.arange(2)
ax.bar(x, [91.2, 1.0], width=0.5, color=["#d9d9d9", "white"], edgecolor=INK, linewidth=0.8)
for xi, v in zip(x, [91.2, 1.0]):
    ax.text(xi, v + 2.5, f"{v:.1f}", ha="center", va="bottom", fontsize=6.6)
ax.set_xticks(x)
ax.set_xticklabels(["with", "deleted"], fontsize=7)
ax.set_ylim(0, 108)
ax.set_ylabel("abstention (%)", fontsize=7)
ax.set_title("(b) instruction", fontsize=7.8, pad=5, loc="left", style="italic")
ax.spines[["top", "right"]].set_visible(False)

# ---------- (c) pool ----------
ax = axs[2]
x = np.arange(2)
ax.bar(x, [100, 59], width=0.5, color=["#d9d9d9", "white"], edgecolor=INK, linewidth=0.8)
for xi, v in zip(x, [100, 59]):
    ax.text(xi, v + 2.5, f"{v:.0f}", ha="center", va="bottom", fontsize=6.6)
ax.set_xticks(x)
ax.set_xticklabels(["pool", "open"], fontsize=7)
ax.set_ylim(0, 118)
ax.set_ylabel("gold in top 10 (%)", fontsize=7)
ax.set_title("(c) pool", fontsize=7.8, pad=5, loc="left", style="italic")
ax.spines[["top", "right"]].set_visible(False)

# ---------- (d) metric ----------
ax = axs[3]
levels = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
risk = np.array([0.98, 0.90, 0.83, 0.76, 0.70])
ax.step(levels, risk, where="post", color=INK, lw=0.9)
ax.plot(levels, risk, color=ACC, lw=0.9, ls=(0, (3, 2)), marker="o", ms=2.4,
        markerfacecolor="white", markeredgewidth=0.7)
ax.fill_between(levels, risk, step="post", facecolor="0.85", edgecolor="none")
ax.set_xlim(0, 1)
ax.set_ylim(0.6, 1.02)
ax.set_xticks([0, 0.5, 1.0])
ax.set_yticks([0.7, 0.85, 1.0])
ax.set_xlabel("coverage", fontsize=7)
ax.set_ylabel("risk", fontsize=7)
ax.set_title("(d) metric", fontsize=7.8, pad=5, loc="left", style="italic")
ax.text(0.03, 0.64, "solid: step\ndashed: trapezoid", fontsize=5.6, color=GREY,
        va="bottom", linespacing=1.35)
ax.spines[["top", "right"]].set_visible(False)

# ---------- magnitude strip ----------
fig.text(0.5, 0.155,
         r"$46$ pts template        $77\text{–}93$ pts instruction        "
         r"$2$ vs $5.5$ pts pool        $0.131$ AURC metric",
         ha="center", fontsize=7.6, color=INK)

fig.text(0.5, 0.075,
         "four choices made before any method is, each larger than the effects methods report",
         ha="center", fontsize=8.2, color=GREY, style="italic")

fig.savefig(OUT, dpi=400, facecolor="white")
print("wrote", OUT)


