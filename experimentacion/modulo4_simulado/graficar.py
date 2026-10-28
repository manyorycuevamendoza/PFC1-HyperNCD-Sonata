"""Figuras de la simulacion del Modulo 4 (lee lo que deja simular_modulo4.py).

    curvas.png         mIoU de las clases novedosas de la linea base (f_geo) y
                       de la propuesta (f_ssl) en cada escenario
    escena.png         escena de referencia: verdad de terreno, linea base y
                       propuesta, desde la misma camara
    escena_3d.html     lo mismo en 3D interactivo, con un boton por vista

Uso:
    python experimentacion/modulo4_simulado/graficar.py --dir experimentacion/modulo4_simulado/resultados
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TITULOS = {
    "calidad_sonata": ("Calidad de Sonata", "separación de las clases nuevas en $f_{ssl}$"),
    "alpha": ("Peso de la geometría $\\alpha$", "$\\alpha$ en $S_b$"),
    "M": ("Vecinos por hiperarista $M$", "$M$"),
    "desbalance": ("Desbalance de clases", "puntos de 'pilar' respecto de 'columna'"),
    "ruido_xyz": ("Ruido en las coordenadas", "desviación en metros"),
}
NOVEDOSAS = ["columna", "pilar", "boveda"]
COLOR = {"base": "#c0392b", "propuesta": "#2471a3"}
PALETA = np.array([[0.85, 0.85, 0.85], [0.6, 0.6, 0.6],          # suelo, muro
                   [0.90, 0.49, 0.13], [0.16, 0.50, 0.73], [0.15, 0.68, 0.38]])


def curvas(d):
    filas = list(csv.DictReader(open(d / "resultados.csv")))
    fig, ejes = plt.subplots(1, len(TITULOS), figsize=(4.2 * len(TITULOS), 3.8))
    for ax, (esc, (tit, xl)) in zip(ejes, TITULOS.items()):
        for ver, nombre in (("base", "Línea base, $f_{geo}$"), ("propuesta", "Propuesta, $f_{ssl}$")):
            f = [r for r in filas if r["escenario"] == esc and r["version"] == ver]
            x = [float(r["valor"]) for r in f]
            y = np.array([float(r["miou"]) for r in f])
            e = np.array([float(r["miou_std"]) for r in f])
            ax.plot(x, y, "o-", color=COLOR[ver], label=nombre)
            ax.fill_between(x, y - e, y + e, color=COLOR[ver], alpha=0.15)
        ax.set_title(tit, fontsize=11)
        ax.set_xlabel(xl, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
    ejes[0].set_ylabel("mIoU de las clases nuevas")
    ejes[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(d / "curvas.png", dpi=150)


def emparejar(verdad, pred):
    """Renombra las clases descubiertas para que coincidan con la verdad."""
    from scipy.optimize import linear_sum_assignment
    K = len(NOVEDOSAS)
    conf = np.zeros((K, K))
    ok = pred >= 0
    np.add.at(conf, (verdad[ok], pred[ok]), 1)
    f, c = linear_sum_assignment(-conf)
    mapa = {q: v for v, q in zip(f, c)}
    return np.array([mapa.get(p, -1) for p in pred])


def escena(d):
    z = np.load(d / "escena_referencia.npz")
    ref = json.loads((d / "referencia.json").read_text())
    xyz, cls, nov = z["xyz"], z["cls"], z["nov"]
    verdad = cls[nov] - 2
    vistas = [("Verdad de terreno", verdad),
              (f"Línea base, mIoU {ref['base']['miou']:.2f}", emparejar(verdad, z["pred_base"])),
              (f"Propuesta, mIoU {ref['propuesta']['miou']:.2f}",
               emparejar(verdad, z["pred_propuesta"]))]
    # la escena es UNA repeticion; el promedio sale del csv
    filas = list(csv.DictReader(open(d / "resultados.csv")))
    prom = {r["version"]: float(r["miou"]) for r in filas
            if r["escenario"] == "calidad_sonata" and float(r["valor"]) == 1.5}
    nota = (f"Repetición 1 de 5, calidad de Sonata 1,5. Promedio de las 5 repeticiones, "
            f"línea base {prom['base']:.2f} y propuesta {prom['propuesta']:.2f}".replace(".", ",", 0))
    conocidos = np.flatnonzero(cls < 2)
    rng = np.random.default_rng(0)
    fondo = rng.choice(conocidos, min(4000, len(conocidos)), replace=False)

    fig = plt.figure(figsize=(15, 5))
    for i, (tit, et) in enumerate(vistas, 1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        ax.scatter(*xyz[fondo].T, c=PALETA[cls[fondo]], s=0.3, alpha=0.3, linewidths=0)
        col = np.where(et[:, None] >= 0, PALETA[np.clip(et, 0, None) + 2], [[0, 0, 0]])
        ax.scatter(*xyz[nov].T, c=col, s=0.6, linewidths=0)
        ax.set_title(tit, fontsize=12)
        ax.view_init(elev=28, azim=-62)
        ax.set_box_aspect(np.ptp(xyz, axis=0))
        ax.set_axis_off()
    manos = [plt.Line2D([], [], marker="o", ls="", color=PALETA[k + 2], label=n)
             for k, n in enumerate(NOVEDOSAS)]
    fig.legend(handles=manos, loc="lower center", ncol=3, fontsize=11)
    fig.suptitle(nota, fontsize=11, y=0.99)
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    fig.savefig(d / "escena.png", dpi=150)

    try:
        import plotly.graph_objects as go
    except ImportError:
        return
    hexa = lambda c: [f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}" for r, g, b in c]
    trazas = [go.Scatter3d(x=xyz[fondo, 0], y=xyz[fondo, 1], z=xyz[fondo, 2], mode="markers",
                           marker=dict(size=1, color=hexa(PALETA[cls[fondo]])),
                           hoverinfo="skip", name="conocidas")]
    for k, (tit, et) in enumerate(vistas):
        col = np.where(et[:, None] >= 0, PALETA[np.clip(et, 0, None) + 2], [[0, 0, 0]])
        trazas.append(go.Scatter3d(x=xyz[nov, 0], y=xyz[nov, 1], z=xyz[nov, 2], mode="markers",
                                   marker=dict(size=1.5, color=hexa(col)), hoverinfo="skip",
                                   name=tit, visible=(k == 0)))
    botones = [dict(label=t, method="update",
                    args=[{"visible": [True] + [j == k for j in range(3)]}])
               for k, (t, _) in enumerate(vistas)]
    fig = go.Figure(trazas)
    fig.update_layout(
        title="Módulo 4 simulado: naranja columna, azul pilar, verde bóveda",
        showlegend=False, height=800, uirevision="fija",
        scene=dict(aspectmode="data", xaxis_visible=False, yaxis_visible=False,
                   zaxis_visible=False),
        updatemenus=[dict(type="buttons", direction="right", x=0.5, y=1.0,
                          xanchor="center", yanchor="bottom", buttons=botones)],
        margin=dict(l=0, r=0, t=90, b=0))
    fig.write_html(d / "escena_3d.html", include_plotlyjs="cdn")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="experimentacion/modulo4_simulado/resultados")
    d = Path(ap.parse_args().dir)
    curvas(d)
    escena(d)
    print(f"[ok] {d}")


if __name__ == "__main__":
    main()
