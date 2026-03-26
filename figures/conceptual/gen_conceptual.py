"""Conceptual curves for poster: ANN vs SNN noise robustness hypothesis"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).parent
x = np.linspace(0, 5, 100)  # severity 0-5

# Shared style
plt.rcParams.update({
    "font.size": 13,
    "axes.linewidth": 1.2,
    "lines.linewidth": 2.5,
})

ANN_COLOR = "#E74C3C"
SNN_COLOR = "#2E86C1"
SNN_REAL_COLOR = "#95A5A6"


# === VERSION 1: Classic hypothesis — ANN fragile, SNN robust ===
fig, ax = plt.subplots(figsize=(7, 4.5))

ann = 0.85 * np.exp(-0.45 * x)
snn = 0.65 * np.exp(-0.12 * x)

ax.plot(x, ann, color=ANN_COLOR, label="ANN (frame-based)")
ax.plot(x, snn, color=SNN_COLOR, label="SNN (event-based)")
ax.fill_between(x, snn, ann, alpha=0.08, color=SNN_COLOR)

ax.set_xlabel("Noise Severity", fontsize=14)
ax.set_ylabel("Detection Performance (mAP)", fontsize=14)
ax.set_xlim(0, 5)
ax.set_ylim(0, 1.0)
ax.set_xticks([0, 1, 2, 3, 4, 5])
ax.legend(fontsize=12, loc="upper right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(OUT / "v1_hypothesis.png", dpi=200, facecolor="white")
plt.savefig(OUT / "v1_hypothesis.pdf", facecolor="white")
plt.close()
print("v1: classic hypothesis")


# === VERSION 2: With "reality" third curve (SNN near zero) ===
fig, ax = plt.subplots(figsize=(7, 4.5))

ann = 0.85 * np.exp(-0.45 * x)
snn_expected = 0.65 * np.exp(-0.12 * x)
snn_reality = 0.01 + 0.005 * np.random.RandomState(42).randn(len(x))
snn_reality = np.clip(snn_reality, 0, 0.03)

ax.plot(x, ann, color=ANN_COLOR, label="ANN (observed)")
ax.plot(x, snn_expected, color=SNN_COLOR, linestyle="--", alpha=0.5, label="SNN (expected)")
ax.plot(x, snn_reality, color=SNN_REAL_COLOR, label="SNN (observed)", linewidth=2)

ax.annotate("Robustness\ngap?", xy=(3, 0.15), fontsize=11, color=SNN_COLOR,
            fontstyle="italic", ha="center", alpha=0.6)
ax.annotate("Near zero", xy=(2.5, 0.04), fontsize=10, color=SNN_REAL_COLOR,
            fontstyle="italic", ha="center")

ax.set_xlabel("Noise Severity", fontsize=14)
ax.set_ylabel("Detection Performance (mAP)", fontsize=14)
ax.set_xlim(0, 5)
ax.set_ylim(-0.02, 1.0)
ax.set_xticks([0, 1, 2, 3, 4, 5])
ax.legend(fontsize=11, loc="upper right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(OUT / "v2_hypothesis_vs_reality.png", dpi=200, facecolor="white")
plt.savefig(OUT / "v2_hypothesis_vs_reality.pdf", facecolor="white")
plt.close()
print("v2: hypothesis vs reality")


# === VERSION 3: Multiple noise types, ANN only ===
fig, ax = plt.subplots(figsize=(7, 4.5))

noises = {
    "Shot noise":     (0.85, 0.20, "#E67E22"),
    "Motion blur":    (0.85, 0.25, "#F39C12"),
    "Gaussian noise": (0.85, 0.40, "#E74C3C"),
    "Snow":           (0.85, 0.55, "#3498DB"),
    "Fog":            (0.85, 1.80, "#8E44AD"),
}

for name, (start, decay, color) in noises.items():
    y = start * np.exp(-decay * x)
    ax.plot(x, y, color=color, label=name)

ax.set_xlabel("Noise Severity", fontsize=14)
ax.set_ylabel("Detection Performance (mAP)", fontsize=14)
ax.set_xlim(0, 5)
ax.set_ylim(0, 1.0)
ax.set_xticks([0, 1, 2, 3, 4, 5])
ax.legend(fontsize=10, loc="upper right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.annotate("Graceful", xy=(4.2, 0.32), fontsize=9, color="#E67E22", fontstyle="italic")
ax.annotate("Catastrophic", xy=(1.8, 0.03), fontsize=9, color="#8E44AD", fontstyle="italic")

plt.tight_layout()
plt.savefig(OUT / "v3_ann_noise_types.png", dpi=200, facecolor="white")
plt.savefig(OUT / "v3_ann_noise_types.pdf", facecolor="white")
plt.close()
print("v3: ANN noise types")


# === VERSION 4: Side by side — hypothesis vs findings ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

# Left: hypothesis
ann = 0.85 * np.exp(-0.45 * x)
snn = 0.65 * np.exp(-0.12 * x)
ax1.plot(x, ann, color=ANN_COLOR, label="ANN")
ax1.plot(x, snn, color=SNN_COLOR, label="SNN")
ax1.fill_between(x, snn, ann, alpha=0.08, color=SNN_COLOR)
ax1.set_title("Hypothesis", fontsize=14, fontweight="bold")
ax1.set_xlabel("Noise Severity", fontsize=13)
ax1.set_ylabel("mAP", fontsize=13)
ax1.set_xlim(0, 5); ax1.set_ylim(0, 1.0)
ax1.set_xticks([0, 1, 2, 3, 4, 5])
ax1.legend(fontsize=11)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# Right: findings
ann_real = 0.85 * np.exp(-0.45 * x)
snn_real = np.full_like(x, 0.01) + 0.003 * np.sin(x * 2)
ax2.plot(x, ann_real, color=ANN_COLOR, label="ANN")
ax2.plot(x, snn_real, color=SNN_COLOR, label="SNN")
ax2.set_title("Observed", fontsize=14, fontweight="bold")
ax2.set_xlabel("Noise Severity", fontsize=13)
ax2.set_ylabel("mAP", fontsize=13)
ax2.set_xlim(0, 5); ax2.set_ylim(0, 1.0)
ax2.set_xticks([0, 1, 2, 3, 4, 5])
ax2.legend(fontsize=11)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

ax2.annotate("SNN baseline\ntoo low to\nmeasure robustness",
             xy=(2.5, 0.08), fontsize=10, fontstyle="italic", color="#666", ha="center")

plt.tight_layout()
plt.savefig(OUT / "v4_hypothesis_vs_findings.png", dpi=200, facecolor="white")
plt.savefig(OUT / "v4_hypothesis_vs_findings.pdf", facecolor="white")
plt.close()
print("v4: side by side")


# === VERSION 5: Clean minimal — just the hypothesis for poster intro ===
fig, ax = plt.subplots(figsize=(6, 4))

ann = 0.85 * np.exp(-0.45 * x)
snn = 0.65 * np.exp(-0.12 * x)

ax.plot(x, ann, color=ANN_COLOR, linewidth=3, label="Frame-based (ANN)")
ax.plot(x, snn, color=SNN_COLOR, linewidth=3, label="Event-based (SNN)")

ax.annotate("", xy=(3.5, 0.65 * np.exp(-0.12 * 3.5)), xytext=(3.5, 0.85 * np.exp(-0.45 * 3.5)),
            arrowprops=dict(arrowstyle="<->", color="#333", lw=1.5))
ax.text(3.7, 0.25, "Robustness\ngap", fontsize=11, color="#333", fontstyle="italic")

ax.set_xlabel("Noise Severity", fontsize=14)
ax.set_ylabel("mAP", fontsize=14)
ax.set_xlim(0, 5); ax.set_ylim(0, 1.0)
ax.set_xticks([]); ax.set_yticks([])
ax.legend(fontsize=12, loc="upper right", framealpha=0.9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(OUT / "v5_minimal_hypothesis.png", dpi=200, facecolor="white")
plt.savefig(OUT / "v5_minimal_hypothesis.pdf", facecolor="white")
plt.close()
print("v5: minimal hypothesis")

print("\nAll done!")
