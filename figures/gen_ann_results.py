"""ANN noise degradation results — real data from zurich_city_14_c evaluation"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).parent

plt.rcParams.update({"font.size": 12, "axes.linewidth": 1.2, "lines.linewidth": 2.2})

# Latest model (subset8 balanced gray) on zurich_city_14_c — full s1-s5
CLEAN = 0.1744
DATA = {
    "Shot noise":     [0.1636, 0.1546, 0.1294, 0.1029, 0.0803],
    "Motion blur":    [0.1663, 0.1449, 0.1207, 0.0979, 0.0679],
    "Glass blur":     [0.1570, 0.1225, 0.0782, 0.0638, 0.0538],
    "Gaussian noise": [0.1482, 0.1208, 0.0871, 0.0423, 0.0101],
    "Impulse noise":  [0.1221, 0.1003, 0.0906, 0.0477, 0.0303],
    "Snow":           [0.1051, 0.0018, 0.0283, 0.0191, 0.0002],
    "Frost":          [0.0506, 0.0000, 0.0000, 0.0000, 0.0303],
    "Fog":            [0.0000, 0.0114, 0.0001, 0.0000, 0.0000],
}

sevs = [1, 2, 3, 4, 5]

COLORS = {
    "Shot noise": "#2ECC71",
    "Motion blur": "#3498DB",
    "Glass blur": "#1ABC9C",
    "Gaussian noise": "#E74C3C",
    "Impulse noise": "#E67E22",
    "Snow": "#9B59B6",
    "Frost": "#8E44AD",
    "Fog": "#34495E",
}


# === FIGURE 1: Line plot — degradation curves ===
fig, ax = plt.subplots(figsize=(8, 5))

ax.axhline(CLEAN, color="#888", linestyle=":", linewidth=1, alpha=0.7, label=f"Clean baseline ({CLEAN:.3f})")

for name, vals in DATA.items():
    ax.plot([0] + sevs, [CLEAN] + vals, color=COLORS[name], marker="o", markersize=4, label=name)

ax.set_xlabel("Noise Severity", fontsize=13)
ax.set_ylabel("mAP@0.5", fontsize=13)
ax.set_xlim(-0.2, 5.2)
ax.set_ylim(-0.005, 0.22)
ax.set_xticks([0, 1, 2, 3, 4, 5])
ax.set_xticklabels(["Clean", "1", "2", "3", "4", "5"])
ax.legend(fontsize=9, loc="upper right", ncol=2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(OUT / "ann_degradation_curves.png", dpi=200, facecolor="white")
plt.savefig(OUT / "ann_degradation_curves.pdf", facecolor="white")
plt.close()
print("1: degradation curves")


# === FIGURE 2: Heatmap ===
fig, ax = plt.subplots(figsize=(8, 5))

names = list(DATA.keys())
matrix = np.array([[CLEAN] + v for v in DATA.values()])  # rows=noise, cols=severity

im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=0.20)
ax.set_xticks(range(6))
ax.set_xticklabels(["Clean", "S1", "S2", "S3", "S4", "S5"])
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names)

# Annotate cells
for i in range(len(names)):
    for j in range(6):
        val = matrix[i, j]
        color = "white" if val < 0.06 else "black"
        ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9, color=color, fontweight="bold")

ax.set_xlabel("Noise Severity", fontsize=13)
cb = plt.colorbar(im, ax=ax, shrink=0.8)
cb.set_label("mAP@0.5", fontsize=11)

plt.tight_layout()
plt.savefig(OUT / "ann_degradation_heatmap.png", dpi=200, facecolor="white")
plt.savefig(OUT / "ann_degradation_heatmap.pdf", facecolor="white")
plt.close()
print("2: heatmap")


# === FIGURE 3: Grouped bar — s3 comparison across noise types ===
fig, ax = plt.subplots(figsize=(9, 4.5))

names_sorted = sorted(DATA.keys(), key=lambda n: DATA[n][2], reverse=True)  # sort by s3
s3_vals = [DATA[n][2] for n in names_sorted]
colors = [COLORS[n] for n in names_sorted]

bars = ax.bar(range(len(names_sorted)), s3_vals, color=colors, edgecolor="white", linewidth=0.5)
ax.axhline(CLEAN, color="#888", linestyle=":", linewidth=1, alpha=0.7)
ax.text(len(names_sorted)-0.5, CLEAN+0.003, f"Clean: {CLEAN:.3f}", fontsize=9, color="#888", ha="right")

ax.set_xticks(range(len(names_sorted)))
ax.set_xticklabels(names_sorted, rotation=30, ha="right", fontsize=10)
ax.set_ylabel("mAP@0.5", fontsize=13)
ax.set_title("ANN Detection Performance at Severity 3", fontsize=13, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

for bar, val in zip(bars, s3_vals):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.003, f"{val:.3f}", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig(OUT / "ann_s3_comparison.png", dpi=200, facecolor="white")
plt.savefig(OUT / "ann_s3_comparison.pdf", facecolor="white")
plt.close()
print("3: s3 bar comparison")


# === FIGURE 4: Retention rate (% of clean mAP retained) ===
fig, ax = plt.subplots(figsize=(8, 5))

ax.axhline(100, color="#888", linestyle=":", linewidth=1, alpha=0.5)

for name, vals in DATA.items():
    retention = [v / CLEAN * 100 for v in vals]
    ax.plot(sevs, retention, color=COLORS[name], marker="o", markersize=4, label=name)

ax.set_xlabel("Noise Severity", fontsize=13)
ax.set_ylabel("mAP Retention (%)", fontsize=13)
ax.set_xlim(0.8, 5.2)
ax.set_ylim(-5, 110)
ax.set_xticks(sevs)
ax.legend(fontsize=9, loc="upper right", ncol=2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(OUT / "ann_retention_curves.png", dpi=200, facecolor="white")
plt.savefig(OUT / "ann_retention_curves.pdf", facecolor="white")
plt.close()
print("4: retention curves")


print("\nAll ANN figures done!")
