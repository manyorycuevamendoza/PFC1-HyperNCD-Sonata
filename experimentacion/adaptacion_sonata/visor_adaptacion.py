"""Visor 3D (un .html) para ver como cambia PL13 entre Sonata ORIGINAL y ADAPTADO.

Botones (la camara no se mueve al cambiar de vista):
  Fila 1 (Modulo 3):  Color real | PCA original | PCA adaptado | Cambio del descriptor |
                      k-means original | k-means adaptado
  Fila 2 (Modulo 4):  grupos K=4 y K=8, con f_ssl original y con f_ssl adaptado

Para que se puedan comparar a ojo, los grupos del adaptado se REORDENAN para que el mismo color
signifique, en lo posible, el mismo grupo del original (asignacion hungara sobre la tabla de
cruce). Los colores de la PCA, en cambio, NO son comparables entre original y adaptado.

Uso (raiz del repo):
    python experimentacion/adaptacion_sonata/visor_adaptacion.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from matplotlib import cm
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import MiniBatchKMeans

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from comparar_m3 import pca_info, rgb as norm_rgb  # noqa: E402

RAIZ = AQUI.parents[1]
DATOS = RAIZ / "experimentacion" / "datos" / "modulo3_PL13"

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--npy", default=str(DATOS / "PL13_SCN0001.npy"))
ap.add_argument("--original", default=str(DATOS / "PL13_feats.npz"))
ap.add_argument("--adaptado", default=str(DATOS / "PL13_feats_adaptado.npz"))
ap.add_argument("--m4-original", default=str(AQUI / "resultados" / "m4_original" / "grupos_PL13.npz"))
ap.add_argument("--m4-adaptado", default=str(AQUI / "resultados" / "m4_adaptado" / "grupos_PL13.npz"))
ap.add_argument("--salida", default=str(AQUI / "resultados" / "visor" / "visor_adaptacion_PL13.html"))
ap.add_argument("--puntos", type=int, default=90_000)
ap.add_argument("--k", type=int, default=8)
ap.add_argument("--tam", type=float, default=1.6)
ap.add_argument("--camara", default="1.25,-1.25,0.95")
a = ap.parse_args()

PAL = np.array(["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf"])
GRIS = "#d0d0d0"


def hexa(c):
    c = (np.clip(c, 0, 1) * 255).astype(int)
    return [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in c]


def alinear(ref, otro, k):
    """Renombra los grupos de `otro` para que coincidan lo mas posible con `ref`."""
    cruce = np.zeros((k, k))
    np.add.at(cruce, (ref, otro), 1)
    fila, col = linear_sum_assignment(-cruce)
    mapa = np.zeros(k, int)
    mapa[col] = fila
    return mapa[otro]


rng = np.random.default_rng(0)
S = np.load(a.npy, mmap_mode="r")
zo, za = np.load(a.original), np.load(a.adaptado)
n = len(S)
sel = np.sort(rng.choice(n, min(n, a.puntos), replace=False))
xyz = np.asarray(S[sel, :3], np.float64).round(3)
color = np.clip(np.asarray(S[sel, 3:6], np.float32) / 255 * 2.2, 0, 1)

fo = zo["feat"][zo["inverse"][sel]].astype(np.float32)
fa = za["feat"][za["inverse"][sel]].astype(np.float32)
_, mo, vo = pca_info(zo["feat"].astype(np.float32), rng)
_, ma, va = pca_info(za["feat"].astype(np.float32), rng)
co, ca = fo - mo, fa - ma
cambio = (co * ca).sum(1) / (np.linalg.norm(co, axis=1) * np.linalg.norm(ca, axis=1) + 1e-8)
no = co / (np.linalg.norm(co, axis=1, keepdims=True) + 1e-8)
na = ca / (np.linalg.norm(ca, axis=1, keepdims=True) + 1e-8)
lo = MiniBatchKMeans(a.k, random_state=0, n_init=3, batch_size=8192).fit_predict(no)
la = alinear(lo, MiniBatchKMeans(a.k, random_state=0, n_init=3, batch_size=8192).fit_predict(na), a.k)

vistas = {
    "Color real": hexa(color),
    "PCA original": hexa(norm_rgb(co @ vo.T)),
    "PCA adaptado": hexa(norm_rgb(ca @ va.T)),
    "Cambio del descriptor": hexa(cm.viridis(np.clip(cambio, 0, 1))[:, :3]),
    f"k-means K={a.k} original": PAL[lo % 8].tolist(),
    f"k-means K={a.k} adaptado": PAL[la % 8].tolist(),
}

# Modulo 4: etiqueta por punto original (solo los "novedosos" tienen grupo; el resto va en gris)
g_o, g_a = np.load(a.m4_original), np.load(a.m4_adaptado)
m4 = {}
for K in (4, 8):
    etiq = {}
    for nom, g in (("orig", g_o), ("adapt", g_a)):
        lab = np.full(n, -1, int)
        lab[g["idx_puntos"]] = g[f"K{K}_M4"][g["region"]]
        etiq[nom] = lab[sel]
    ok = (etiq["orig"] >= 0) & (etiq["adapt"] >= 0)
    ad = etiq["adapt"].copy()
    ad[ok] = alinear(etiq["orig"][ok], etiq["adapt"][ok], K) if K <= 8 else ad[ok]
    for nom, lab in (("original", etiq["orig"]), ("adaptado", ad)):
        col = np.where(lab >= 0, PAL[np.clip(lab, 0, 7) % 8], GRIS)
        m4[f"M4 K={K} {nom}"] = col.tolist()

todas = {**vistas, **m4}
nombres = list(todas)
fig = go.Figure(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers",
                             marker=dict(size=a.tam, color=todas[nombres[0]]), hoverinfo="skip"))
def botones(lista):
    return [dict(label=t, method="restyle", args=[{"marker.color": [todas[t]]}]) for t in lista]
lim = np.array([np.percentile(S[:, k], [0.5, 99.5]) for k in range(3)])
lado = lim[:, 1] - lim[:, 0]
fig.update_layout(
    title=dict(text="PL13: Sonata ORIGINAL vs ADAPTADO a la Catedral · gira con el mouse · 📷 baja el PNG<br>"
                    "<sup>Los grupos del adaptado se reordenan para parecerse al original (mismo color ≈ mismo grupo). "
                    "Los colores de la PCA no son comparables. Gris = punto sin grupo en M4.</sup>", y=0.985),
    showlegend=False, height=880, uirevision="fija",
    scene=dict(camera=dict(eye=dict(zip("xyz", map(float, a.camara.split(",")))), up=dict(x=0, y=0, z=1)),
               aspectmode="manual", aspectratio=dict(zip("xyz", (lado / lado.max()).tolist())),
               xaxis=dict(visible=False, range=lim[0].tolist()),
               yaxis=dict(visible=False, range=lim[1].tolist()),
               zaxis=dict(visible=False, range=lim[2].tolist())),
    updatemenus=[
        dict(type="buttons", direction="right", x=0.5, y=1.0, xanchor="center", yanchor="bottom",
             buttons=botones(list(vistas)), pad=dict(b=36)),
        dict(type="buttons", direction="right", x=0.5, y=1.0, xanchor="center", yanchor="bottom",
             buttons=botones(list(m4))),
    ],
    margin=dict(l=0, r=0, t=170, b=20))
sal = Path(a.salida)
sal.parent.mkdir(parents=True, exist_ok=True)
fig.write_html(sal, include_plotlyjs="cdn", config=dict(displaylogo=False,
               toImageButtonOptions=dict(format="png", filename="PL13_adaptacion", scale=2)))
print(sal, sal.stat().st_size // 2**20, "MB")
