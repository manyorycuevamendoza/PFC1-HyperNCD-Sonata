"""Figura cualitativa en 3D: aciertos (verde azulado) y errores (rojo) de un baseline y de M4, misma camara.

Cada grupo recibe la clase del emparejamiento 1 a 1 (Hungarian, sin la clase `otro`), igual que
experimentacion/arch_entrenamiento/graficar_escena.py. Los puntos de `otro` no se dibujan ni cuentan.
Camara en perspectiva escrita a mano (acimut, elevacion, distancia) para repetirla igual en ambos paneles.

Uso (desde la raiz del repo):
    python experimentacion/escena_capilla_etiquetada/figura_cualitativa.py \
        --grupos <m4_original/grupos_PL13.npz> --npy <capilla_etiquetada.npy> \
        --etiquetas <capilla_etiquetada_etiquetas.npy> --salida documentacion/poster/fig/cualitativa
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.optimize import linear_sum_assignment

ap = argparse.ArgumentParser()
ap.add_argument("--grupos", required=True)
ap.add_argument("--npy", required=True)
ap.add_argument("--etiquetas", required=True)
ap.add_argument("--salida", required=True, help="prefijo sin extension; escribe .png y .pdf")
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--base", default="M4_base")
ap.add_argument("--nuestro", default="M4")
ap.add_argument("--acimut", type=float, default=35.0)
ap.add_argument("--elev", type=float, default=24.0)
ap.add_argument("--dist", type=float, default=2.6, help="distancia de la camara en radios de la escena")
ap.add_argument("--semilla", type=int, default=0)
a = ap.parse_args()

OK, MAL = np.array([0.0, 0.62, 0.55]), np.array([0.86, 0.10, 0.10])   # verde azulado e rojo intenso

z = np.load(a.grupos)
S = np.load(a.npy, mmap_mode="r")
y_all = np.load(a.etiquetas).astype(int)
idx, reg = z["idx_puntos"].astype(int), z["region"].astype(int)
y = y_all[idx]
valido = (reg >= 0) & (y < 5)
xyz = np.asarray(S[idx, :3], dtype=np.float64)
nrm = np.asarray(S[idx, 6:9], dtype=np.float64)


def predecir(clave):
    g = np.full(len(idx), -1, dtype=int)
    g[reg >= 0] = z[clave][reg[reg >= 0]]
    m = (g >= 0) & (y < 5)
    T = np.zeros((a.K, 5))
    np.add.at(T, (g[m], y[m]), 1)
    f, c = linear_sum_assignment(-T)
    pc = np.full(len(idx), -1, dtype=int)
    for fi, ci in zip(f, c):
        pc[g == fi] = ci
    ious = []
    for c_ in range(5):
        tp = np.sum((pc == c_) & (y == c_) & m)
        fp = np.sum((pc == c_) & (y != c_) & (y < 5) & (g >= 0))
        fn = np.sum((pc != c_) & (y == c_) & m)
        ious.append(tp / max(tp + fp + fn, 1))
    exact = np.sum((pc == y) & valido) / valido.sum()
    return pc, 100 * float(np.mean(ious)), 100 * float(exact)


res = {n: predecir(f"K{a.K}_{n}") for n in (a.base, a.nuestro)}

# ---- camara en perspectiva ----
P = xyz[valido]
centro = (P.min(0) + P.max(0)) / 2
R = np.linalg.norm(P.max(0) - P.min(0)) / 2
az, el = np.radians(a.acimut), np.radians(a.elev)
cam = centro + a.dist * R * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
f_ = (centro - cam) / np.linalg.norm(centro - cam)
r_ = np.cross(f_, [0, 0, 1]); r_ /= np.linalg.norm(r_)
u_ = np.cross(r_, f_)


def proyectar(X):
    d = X - cam
    pz = d @ f_
    return np.c_[(d @ r_) / pz, (d @ u_) / pz], pz


uv, prof = proyectar(xyz)
luz = np.array([-0.4, -0.5, 0.75]); luz /= np.linalg.norm(luz)
sombra = 0.62 + 0.38 * np.abs(nrm @ luz)
sombra *= 1.0 - 0.18 * (prof - prof[valido].min()) / np.ptp(prof[valido])   # mas lejos, algo mas oscuro


def colores(pred):
    ok = pred == y
    c = np.where(ok[:, None], OK, MAL) * sombra[:, None]
    return np.clip(c, 0, 1), ok


rng = np.random.default_rng(a.semilla)
orden = np.where(valido)[0]
orden = orden[np.argsort(-prof[orden])]           # de lejos a cerca (algoritmo del pintor)
C = {n: colores(res[n][0]) for n in res}

# zona de zoom: ventana donde mas empeora el baseline respecto a nuestro modelo
lo, hi = uv[orden].min(0), uv[orden].max(0)
ancho = (hi - lo)
vent = 0.24 * ancho
ventanas = []
for gx in np.linspace(lo[0], hi[0] - vent[0], 24):
    for gy in np.linspace(lo[1], hi[1] - vent[1], 36):
        dentro = (uv[orden, 0] >= gx) & (uv[orden, 0] <= gx + vent[0]) & (uv[orden, 1] >= gy) & (uv[orden, 1] <= gy + vent[1])
        ventanas.append((dentro.sum(), (~C[a.base][1][orden][dentro]).mean() - (~C[a.nuestro][1][orden][dentro]).mean(), (gx, gy)))
nmax = max(v[0] for v in ventanas)
mejor, pos = -1, None
for n_, dif, p_ in ventanas:           # solo ventanas bien pobladas (>= 60 % de la mas densa)
    if n_ >= 0.6 * nmax and dif > mejor:
        mejor, pos = dif, p_
x0, y0 = pos

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
fig = plt.figure(figsize=(7.2, 4.9))
mx, my = lo - 0.03 * ancho, hi + 0.03 * ancho
titulos = {a.base: "Baseline", a.nuestro: "Ours"}
for j, n in enumerate((a.base, a.nuestro)):
    ax = fig.add_axes([0.01 + j * 0.495, 0.17, 0.485, 0.76])
    ax.scatter(uv[orden, 0], uv[orden, 1], c=C[n][0][orden], s=0.9, linewidths=0, marker="o", rasterized=True)
    ax.add_patch(Rectangle((x0, y0), vent[0], vent[1], fill=False, ec="black", lw=1.0))
    ax.set_xlim(mx[0], my[0]); ax.set_ylim(mx[1], my[1]); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(titulos[n], fontsize=11, fontweight="bold", pad=2)
    ins = ax.inset_axes([0.60, 0.0, 0.40, 0.30])
    sel = orden[(uv[orden, 0] >= x0) & (uv[orden, 0] <= x0 + vent[0]) & (uv[orden, 1] >= y0) & (uv[orden, 1] <= y0 + vent[1])]
    ins.scatter(uv[sel, 0], uv[sel, 1], c=C[n][0][sel], s=1.8, linewidths=0, rasterized=True)
    ins.set_xlim(x0, x0 + vent[0]); ins.set_ylim(y0, y0 + vent[1]); ins.set_xticks([]); ins.set_yticks([])
    for s in ins.spines.values():
        s.set_linewidth(1.0)
    fig.text(0.01 + j * 0.495 + 0.2425, 0.105, f"mIoU {res[n][1]:.1f} %   |   exactitud {res[n][2]:.1f} %", ha="center", fontsize=10)

fig.legend(handles=[plt.Line2D([0], [0], marker="o", ls="", color=OK, label="correcto"),
                    plt.Line2D([0], [0], marker="o", ls="", color=MAL, label="error")],
           loc="lower center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.0))
fig.savefig(a.salida + ".png", dpi=300, facecolor="white")
fig.savefig(a.salida + ".pdf", dpi=300, facecolor="white")
print({n: (round(res[n][1], 1), round(res[n][2], 1)) for n in res},
      f"camara: acimut {a.acimut} deg, elevacion {a.elev} deg, distancia {a.dist} R, R={R:.1f} m, centro={np.round(centro, 1)}")
