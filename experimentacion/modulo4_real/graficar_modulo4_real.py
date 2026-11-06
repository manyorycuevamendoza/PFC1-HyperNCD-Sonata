"""Figuras del Modulo 4 real: mapa de grupos por variante y acuerdo entre variantes."""
import argparse, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="experimentacion/resultados/modulo4_real")
ap.add_argument("--k", type=int, default=4)
a = ap.parse_args()
d = Path(a.dir)
z = np.load(d / "grupos_PL13.npz")
js = json.loads((d / "modulo4_real_PL13.json").read_text())["K"][str(a.k)]
cen = z["cen"]
nombres = ["M2", "M3", "M4_base", "M4"]
titulos = {"M2": "M2: f_sem", "M3": "M3: f_ssl", "M4_base": "M4 base: [f_sem; f_geo]",
           "M4": "M4 propuesta: [f_sem; f_ssl]"}
fig, ax = plt.subplots(2, 4, figsize=(18, 8))
for j, n in enumerate(nombres):
    lab = z[f"K{a.k}_{n}"]
    for i, (u, v, tx) in enumerate(((0, 1, "planta X-Y"), (0, 2, "lateral X-Z"))):
        ax[i, j].scatter(cen[:, u], cen[:, v], c=lab, cmap="tab10", vmin=0, vmax=9, s=4)
        ax[i, j].set_aspect("equal"); ax[i, j].set_title(f"{titulos[n]}\n{tx}", fontsize=10)
plt.tight_layout(); plt.savefig(d / f"mapas_K{a.k}.png", dpi=130); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
M = np.eye(4)
for k, v in js["acuerdo"].items():
    x, y = k.split("~"); M[nombres.index(x), nombres.index(y)] = M[nombres.index(y), nombres.index(x)] = v["ari"]
im = ax[0].imshow(M, vmin=0, vmax=1, cmap="viridis")
ax[0].set_xticks(range(4)); ax[0].set_xticklabels(nombres); ax[0].set_yticks(range(4)); ax[0].set_yticklabels(nombres)
for i in range(4):
    for j in range(4): ax[0].text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", color="w")
ax[0].set_title("Acuerdo entre variantes (ARI)")
T = np.array(js["tabla_m2"]["M4"]); T = T[T.sum(1) > 2000]
ax[1].bar(range(len(nombres)), [js[n]["estabilidad_ari"] for n in nombres])
ax[1].set_xticks(range(4)); ax[1].set_xticklabels(nombres); ax[1].set_ylim(0, 1)
ax[1].set_title("Estabilidad entre 3 semillas (ARI)")
plt.tight_layout(); plt.savefig(d / f"acuerdo_K{a.k}.png", dpi=130); plt.close()
print("ok")
