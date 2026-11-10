"""Grafica M2, M3, M4 base y M4 con media y desviacion sobre semillas (lee resultados/referencias_semillas.json)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path(__file__).resolve().parent / "resultados"
d = json.loads((R / "referencias_semillas.json").read_text(encoding="utf-8"))
K = "8"
combos = {"semposs_original": "HyperNCD original\n+ Sonata original", "semposs_adaptado": "HyperNCD original\n+ Sonata adaptado",
          "arch_original": "HyperNCD adaptado\n+ Sonata original", "arch_adaptado": "HyperNCD adaptado\n+ Sonata adaptado"}
mods = [("M2", "M2 (solo f_sem)", "#9aa5b1"), ("M3", "M3 (solo f_ssl)", "#f0b48a"), ("M4_base", "M4 base (f_sem + f_geo)", "#2a7fbf"), ("M4", "M4 (f_sem + f_ssl)", "#e07b39")]
fig, ax = plt.subplots(figsize=(12, 4.8))
w = 0.2
for j, (m, et, col) in enumerate(mods):
    mu = [d["datos"][c][K][m]["media"] for c in combos]
    sd = [d["datos"][c][K][m]["std"] for c in combos]
    lo = [d["datos"][c][K][m]["min"] for c in combos]
    hi = [d["datos"][c][K][m]["max"] for c in combos]
    x = np.arange(len(combos)) + (j - 1.5) * w
    ax.bar(x, mu, w, yerr=sd, capsize=3, label=et, color=col, edgecolor="k", linewidth=0.5)
    ax.scatter(np.r_[x, x], np.r_[lo, hi], s=10, color="k", zorder=3)
ax.set_xticks(range(len(combos)))
ax.set_xticklabels(list(combos.values()))
ax.set_ylabel("mIoU % (1 a 1, sin «otro»)")
ax.set_ylim(0, 70)
ax.set_title(f"Media ± desviación sobre {d['semillas']} semillas (K={K}); los puntos negros son el mínimo y el máximo")
ax.legend(fontsize=8, ncol=2, loc="upper left")
fig.tight_layout()
fig.savefig(R / "referencias_semillas.png", dpi=130)
print("ok")
