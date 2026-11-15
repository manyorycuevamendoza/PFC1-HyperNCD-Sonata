"""Figura de las etapas sobre la escena etiquetada: crudo, Modulo 1 y descriptores de Sonata (PCA).

Solo se dibujan las piezas etiquetadas (arco, columna, cupula, pared, techo). La clase `otro` queda fuera
porque reune todo lo que no se etiqueto y confunde la lectura de la imagen.

Uso (desde la raiz del repo):
    python experimentacion/escena_capilla_etiquetada/graficar_etapas.py --las <carpeta con los .las de listo> \
        --npy <capilla.npy> --etiquetas <etiquetas.npy> --feats <feats_original.npz> --salida <png>
"""
import argparse
import glob
import os
import numpy as np
import laspy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

ap = argparse.ArgumentParser()
ap.add_argument("--las", required=True)
ap.add_argument("--npy", required=True)
ap.add_argument("--etiquetas", required=True)
ap.add_argument("--feats", required=True)
ap.add_argument("--salida", required=True)
ap.add_argument("--n", type=int, default=250000)
a = ap.parse_args()
rng = np.random.default_rng(0)

# crudo, solo piezas etiquetadas
xs, cs = [], []
for f in sorted(glob.glob(os.path.join(a.las, "*.las"))):
    if os.path.basename(f).lower().startswith("otro"):
        continue
    las = laspy.read(f)
    xs.append(np.vstack([las.x, las.y, las.z]).T.astype(np.float64))
    cs.append(np.vstack([las.red, las.green, las.blue]).T.astype(np.float32) / 65535.0)
raw = np.vstack(xs)
rawc = np.vstack(cs)
n_raw = len(raw)

S = np.load(a.npy, mmap_mode="r")
y = np.load(a.etiquetas).astype(int)
lab = np.where(y < 5)[0]
n_m1 = len(lab)
xyz = np.asarray(S[lab, :3], dtype=np.float32)
rgb = np.asarray(S[lab, 3:6], dtype=np.float32) / 255.0

z = np.load(a.feats)
feat = z["feat"]
inv = z["inverse"]
fl = feat[inv[lab]].astype(np.float32)
muestra = rng.choice(len(fl), size=min(100000, len(fl)), replace=False)
mu = fl[muestra].mean(0)
pca = PCA(n_components=3, random_state=0).fit(fl[muestra] - mu)
proy = pca.transform(fl - mu)
lo, hi = np.percentile(proy, 2, axis=0), np.percentile(proy, 98, axis=0)
pcol = np.clip((proy - lo) / (hi - lo), 0, 1)


def claro(c):
    return np.clip(c, 0, 1) ** 0.4


def submuestra(n):
    return rng.choice(n, size=min(a.n, n), replace=False)


i_raw = submuestra(len(raw))
i_m1 = submuestra(len(xyz))
paneles = [
    ("(a) Crudo, piezas etiquetadas\n" + f"{n_raw:,}".replace(",", " ") + " puntos", raw[i_raw], claro(rawc[i_raw])),
    ("(b) Módulo 1\n" + f"{n_m1:,}".replace(",", " ") + " puntos", xyz[i_m1], claro(rgb[i_m1])),
    ("(c) Módulo 3, PCA de $f_{ssl}$\nmismos puntos del Módulo 1", xyz[i_m1], pcol[i_m1]),
]

fig, ejes = plt.subplots(2, 3, figsize=(9.5, 7.4))
for j, (tit, P, col) in enumerate(paneles):
    o = np.argsort(P[:, 2])
    ejes[0, j].scatter(P[o, 0], P[o, 1], c=col[o], s=0.3, linewidths=0, rasterized=True)
    ejes[1, j].scatter(P[o, 0], P[o, 2], c=col[o], s=0.3, linewidths=0, rasterized=True)
    ejes[0, j].set_title(tit, fontsize=10)
    for i in (0, 1):
        ejes[i, j].set_aspect("equal")
        ejes[i, j].set_xticks([])
        ejes[i, j].set_yticks([])
ejes[0, 0].set_ylabel("planta (x, y)")
ejes[1, 0].set_ylabel("lateral (x, z)")
fig.subplots_adjust(top=0.9, bottom=0.02, wspace=0.03, hspace=0.04, left=0.04, right=0.99)
fig.savefig(a.salida, dpi=130)
print(n_raw, n_m1)
