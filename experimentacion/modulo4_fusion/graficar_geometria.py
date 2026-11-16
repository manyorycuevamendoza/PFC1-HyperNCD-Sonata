"""Figura de la geometria de cada clase frente al IoU del mejor modelo (M4 con M2 de ArCH y Sonata original).

El IoU es la media de diez semillas de `resultados/referencias_semillas.json`. Las cuatro medidas geometricas de cada
clase salen de la capilla etiquetada y de los archivos LAS por pieza: tamano (% de la escena), linealidad, puntos con
vecinos de otra clase (%) y dispersion de las normales. Los valores de las tres primeras coinciden con la tabla
`tab:geometria` de la tesis; los de la dispersion de las normales se leyeron de la figura anterior, que no guardaba
los datos de origen.

Uso (desde la raiz del repo):
    python experimentacion/modulo4_fusion/graficar_geometria.py \
        --salida documentacion/tesis_latex/asesoria5/imagenes/caso_grafico_geometria.png
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--json", default=os.path.join(os.path.dirname(__file__), "resultados", "referencias_semillas.json"))
ap.add_argument("--salida", required=True)
a = ap.parse_args()

CLASES = ["Pared", "Techo", "Cúpula", "Arco", "Columna"]
CLAVE = {"Pared": "pared", "Techo": "techo", "Cúpula": "cupula", "Arco": "arco", "Columna": "columna"}

MEDIDAS = [
    ("Tamaño de la clase\n(% de los puntos de la escena)",
     {"Pared": 15.33, "Techo": 6.63, "Cúpula": 9.72, "Arco": 4.23, "Columna": 2.93}),
    ("Linealidad (forma alargada)\n0 a 1",
     {"Pared": 0.194, "Techo": 0.198, "Cúpula": 0.272, "Arco": 0.380, "Columna": 0.351}),
    ("Puntos con vecinos de\notra clase (%)",
     {"Pared": 1.25, "Techo": 0.80, "Cúpula": 0.54, "Arco": 2.50, "Columna": 2.92}),
    ("Dispersión de las normales\n0 a 1",
     {"Pared": 0.175, "Techo": 0.120, "Cúpula": 0.656, "Arco": 0.636, "Columna": 0.444}),
]

with open(a.json, encoding="utf8") as f:
    iou = json.load(f)["datos"]["arch_original"]["8"]["M4"]["iou_por_clase"]
y = {c: iou[CLAVE[c]] for c in CLASES}

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 8.5})
fig, ejes = plt.subplots(2, 2, figsize=(6.1, 4.4), sharey=True)
for ax, (etiqueta, x) in zip(ejes.ravel(), MEDIDAS):
    xs = [x[c] for c in CLASES]
    lo, hi = min(xs), max(xs)
    margen = (hi - lo) * 0.12
    ax.set_xlim(lo - margen, hi + margen)
    ax.scatter(xs, [y[c] for c in CLASES], s=34, color="#1f4e8c", zorder=3)
    for c in CLASES:
        derecha = (x[c] - lo) / (hi - lo) > 0.72
        # la etiqueta del techo va debajo cuando la pared queda justo encima y a la derecha
        debajo = c == "Techo" and abs(x["Pared"] - x["Techo"]) < 0.25 * (hi - lo) and etiqueta[:4] != "Line"
        despl = (4, -11) if debajo else ((-5 if derecha else 5), 4)
        ax.annotate(c, (x[c], y[c]), xytext=despl, textcoords="offset points",
                    ha="right" if (derecha and not debajo) else "left", fontsize=8.5)
    ax.set_xlabel(etiqueta, fontsize=8.5)
    ax.grid(color="#e8e8e8", linewidth=0.7, zorder=0)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.tick_params(labelsize=8.5)
ejes[0, 0].set_ylabel("IoU del mejor modelo (%)", fontsize=8.5)
ejes[1, 0].set_ylabel("IoU del mejor modelo (%)", fontsize=8.5)
ejes[0, 0].set_ylim(25, 76)
fig.subplots_adjust(left=0.10, right=0.965, top=0.985, bottom=0.14, wspace=0.08, hspace=0.55)
fig.savefig(a.salida, dpi=220)
print("listo", a.salida)
