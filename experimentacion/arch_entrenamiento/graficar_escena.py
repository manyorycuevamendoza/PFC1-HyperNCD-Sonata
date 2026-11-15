"""Figura de la escena etiquetada: etiqueta real y agrupamiento de M2, M3, M4 base y M4 (K = 8).

Vista en planta (x, y) y vista lateral (x, z). Cada grupo se pinta con la clase que le asigna el emparejamiento
uno a uno (Hungarian) calculado sin la clase `otro`; los grupos sin pareja salen en gris oscuro.
Una sola semilla (la de la corrida guardada), asi que es una imagen ilustrativa, no una medida.

Uso (desde la raiz del repo):
    python experimentacion/arch_entrenamiento/graficar_escena.py \
        --grupos experimentacion/arch_entrenamiento/datos_visor/m4/salida/capilla_etiquetada/m4_original/grupos_PL13.npz \
        --npy experimentacion/arch_entrenamiento/datos_visor/m1/salida/capilla_etiquetada/capilla_etiquetada.npy \
        --etiquetas experimentacion/arch_entrenamiento/datos_visor/m1/salida/capilla_etiquetada/capilla_etiquetada_etiquetas.npy \
        --salida documentacion/tesis_latex/asesoria5/imagenes/escena_m1_m4.png
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

CLASES = ["arco", "columna", "cúpula", "pared", "techo"]
COLORES = ["#d62728", "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
GRIS_OTRO = "#d9d9d9"
GRIS_SIN = "#4d4d4d"

ap = argparse.ArgumentParser()
ap.add_argument("--grupos", required=True)
ap.add_argument("--npy", required=True)
ap.add_argument("--etiquetas", required=True)
ap.add_argument("--salida", required=True)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--n", type=int, default=250000, help="puntos que se dibujan")
a = ap.parse_args()

z = np.load(a.grupos)
S = np.load(a.npy, mmap_mode="r")
y_all = np.load(a.etiquetas).astype(int)
idx = z["idx_puntos"].astype(int)
reg = z["region"].astype(int)
xyz = np.asarray(S[idx, :3], dtype=np.float32)
y = y_all[idx]                      # 0..4 clases, 5 = otro
validos = reg >= 0

metodos = [("M2", f"K{a.K}_M2"), ("M3", f"K{a.K}_M3"), ("M4 base", f"K{a.K}_M4_base"), ("M4", f"K{a.K}_M4")]
pred, miou = {}, {}
for nombre, clave in metodos:
    g = np.full(len(idx), -1, dtype=int)
    g[validos] = z[clave][reg[validos]]
    m = (g >= 0) & (y < 5)
    T = np.zeros((a.K, 5))
    np.add.at(T, (g[m], y[m]), 1)
    fila, col = linear_sum_assignment(-T)
    mapa = {int(f): int(c) for f, c in zip(fila, col)}
    pc = np.full(len(idx), -1, dtype=int)
    for f, c in mapa.items():
        pc[g == f] = c
    pred[nombre] = pc
    ious = []
    for c in range(5):
        tp = np.sum((pc == c) & (y == c) & m)
        fp = np.sum((pc == c) & (y != c) & (y < 5) & (g >= 0))
        fn = np.sum((pc != c) & (y == c) & m)
        ious.append(tp / max(tp + fp + fn, 1))
    miou[nombre] = 100 * float(np.mean(ious))

rng = np.random.default_rng(0)
etiquetados = np.where(y < 5)[0]          # la clase otro no se dibuja, confunde la lectura
sel = rng.choice(etiquetados, size=min(a.n, len(etiquetados)), replace=False)
sel = sel[np.argsort(xyz[sel, 2])]
P = xyz[sel]


def colores(lbl):
    c = np.empty(len(lbl), dtype=object)
    for k in range(5):
        c[lbl == k] = COLORES[k]
    c[lbl == 5] = GRIS_OTRO
    c[lbl == -1] = GRIS_SIN
    return list(c)


paneles = [("Etiqueta real", colores(y[sel]))]
for nombre, _ in metodos:
    paneles.append((f"{nombre}  (mIoU {miou[nombre]:.1f})", colores(pred[nombre][sel])))

fig, ejes = plt.subplots(2, 5, figsize=(14, 7.2))
for j, (titulo, col) in enumerate(paneles):
    ejes[0, j].scatter(P[:, 0], P[:, 1], c=col, s=0.25, linewidths=0, rasterized=True)
    ejes[0, j].set_title(titulo, fontsize=10)
    ejes[1, j].scatter(P[:, 0], P[:, 2], c=col, s=0.25, linewidths=0, rasterized=True)
    for i in (0, 1):
        ejes[i, j].set_aspect("equal")
        ejes[i, j].set_xticks([])
        ejes[i, j].set_yticks([])
ejes[0, 0].set_ylabel("planta (x, y)")
ejes[1, 0].set_ylabel("lateral (x, z)")
handles = [plt.Line2D([0], [0], marker="o", ls="", color=c, label=n) for n, c in zip(CLASES, COLORES)]
handles += [plt.Line2D([0], [0], marker="o", ls="", color=GRIS_SIN, label="grupo sin clase")]
fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False, fontsize=9)
fig.subplots_adjust(bottom=0.1, top=0.93, wspace=0.02, hspace=0.04, left=0.02, right=0.99)
fig.savefig(a.salida, dpi=130)
print({k: round(v, 1) for k, v in miou.items()})
