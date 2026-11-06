"""Visor 3D de los grupos del Modulo 4 real en PL13, con el mismo estilo que
experimentacion/visor_3d.py (una traza por etapa, camara fija, escala fija)."""
import argparse
from pathlib import Path
import numpy as np
import plotly.graph_objects as go

ap = argparse.ArgumentParser()
ap.add_argument("--npy", default="experimentacion/datos/modulo3_PL13/PL13_SCN0001.npy")
ap.add_argument("--dir", default="experimentacion/resultados/modulo4_real")
ap.add_argument("--puntos", type=int, default=120000)
ap.add_argument("--tam", type=float, default=1.6)
ap.add_argument("--camara", default="1.25,-1.25,0.95")
a = ap.parse_args()
d = Path(a.dir)
z = np.load(d / "grupos_PL13.npz")
S = np.load(a.npy, mmap_mode="r")
idx, reg = z["idx_puntos"], z["region"]
rng = np.random.default_rng(0)
sel = np.sort(rng.choice(len(idx), min(a.puntos, len(idx)), replace=False))
xyz = np.asarray(S[idx[sel], :3], np.float64).round(3)
rgb = np.clip(np.asarray(S[idx[sel], 3:6], np.float32) / 255 * 2.2, 0, 1)
reg = reg[sel]
PAL = np.array(["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf"])
nombres = {"M2": "M2 (f_sem)", "M3": "M3 (f_ssl)", "M4_base": "M4 base [f_sem; f_geo]",
           "M4": "M4 propuesta [f_sem; f_ssl]"}
etapas = [("Color real", [f"rgb({int(r*255)},{int(g*255)},{int(b*255)})" for r, g, b in rgb])]
for K in (4, 8):
    for n, t in nombres.items():
        if f"K{K}_{n}" in z.files:
            etapas.append((f"K={K} {t}", PAL[z[f"K{K}_{n}"][reg] % 8].tolist()))
trazas = [go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers",
                       marker=dict(size=a.tam, color=c), hoverinfo="skip", name=n,
                       visible=(k == 0)) for k, (n, c) in enumerate(etapas)]
botones = [dict(label=n, method="update", args=[{"visible": [e == k for e in range(len(etapas))]}])
           for k, (n, _) in enumerate(etapas)]
lim = np.array([np.percentile(S[:, k], [0.5, 99.5]) for k in range(3)])
lado = lim[:, 1] - lim[:, 0]
fig = go.Figure(trazas)
fig.update_layout(
    title=dict(text="PL13, Módulo 4 real: cada color es un grupo · gira con el mouse · 📷 baja el PNG", y=0.98),
    showlegend=False, height=850, uirevision="fija",
    scene=dict(camera=dict(eye=dict(zip("xyz", map(float, a.camara.split(",")))), up=dict(x=0, y=0, z=1)),
               aspectmode="manual", aspectratio=dict(zip("xyz", (lado / lado.max()).tolist())),
               xaxis=dict(visible=False, range=lim[0].tolist()),
               yaxis=dict(visible=False, range=lim[1].tolist()),
               zaxis=dict(visible=False, range=lim[2].tolist())),
    updatemenus=[dict(type="buttons", direction="right", x=0.5, y=1.0, xanchor="center",
                      yanchor="bottom", buttons=botones)],
    margin=dict(l=0, r=0, t=110, b=30))
sal = d / "visor_modulo4_PL13.html"
fig.write_html(sal, include_plotlyjs="cdn", config=dict(displaylogo=False,
               toImageButtonOptions=dict(format="png", filename="PL13_modulo4", scale=2)))
print(sal, sal.stat().st_size // 2**20, "MB")
