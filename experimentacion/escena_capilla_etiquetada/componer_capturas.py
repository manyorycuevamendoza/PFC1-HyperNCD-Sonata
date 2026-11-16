"""Compone las figuras de M1, M2 y M3 con capturas del visor de la escena (visor_escena_comparacion_m2.html).

Las capturas se sacan del propio visor, con la clase `otro` oculta, y este programa las recorta y las junta con
titulos y leyendas. No recalcula nada, solo muestra lo que dibuja el visor.

Uso (desde la raiz del repo):
    python experimentacion/escena_capilla_etiquetada/componer_capturas.py --capturas <carpeta con los png> --salida <carpeta>
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.image import imread

ap = argparse.ArgumentParser()
ap.add_argument("--capturas", required=True)
ap.add_argument("--salida", required=True)
a = ap.parse_args()

CLASES = [("arco", "#e6194b"), ("columna", "#3cb44b"), ("cúpula", "#0082c8"), ("pared", "#f58231"), ("techo", "#911eb4")]


def cargar(nombre):
    im = imread(os.path.join(a.capturas, nombre))[..., :3]
    return im


def recorte_comun(nombres, margen=12):
    """Recorte que contiene el dibujo de todas las capturas, para que queden con el mismo encuadre."""
    y0, y1, x0, x1 = 10**9, 0, 10**9, 0
    for n in nombres:
        im = cargar(n)
        m = (im < 0.97).any(axis=2)
        ys, xs = np.where(m)
        y0, y1 = min(y0, ys.min()), max(y1, ys.max())
        x0, x1 = min(x0, xs.min()), max(x1, xs.max())
    return max(y0 - margen, 0), y1 + margen, max(x0 - margen, 0), x1 + margen


def figura(paneles, filas, columnas, salida, leyenda, ancho=11.0, alto_panel=4.3, fuente=10, fuente_leyenda=9,
           dpi=130, ncol_leyenda=None, alto_leyenda=0.6, top=0.94, bottom=0.07, hspace=0.12):
    nombres = [p[0] for p in paneles]
    y0, y1, x0, x1 = recorte_comun(nombres)
    fig, ejes = plt.subplots(filas, columnas, figsize=(ancho, alto_panel * filas + alto_leyenda), squeeze=False)
    for ax in ejes.ravel():
        ax.axis("off")
    for k, (nombre, titulo) in enumerate(paneles):
        ax = ejes[k // columnas, k % columnas]
        ax.imshow(cargar(nombre)[y0:y1, x0:x1])
        ax.set_title(titulo, fontsize=fuente)
    h = [plt.Line2D([0], [0], marker="o", ls="", color=c, label=n) for n, c in leyenda]
    if h:
        fig.legend(handles=h, loc="lower center", ncol=ncol_leyenda or len(h), frameon=False, fontsize=fuente_leyenda)
    fig.subplots_adjust(left=0.01, right=0.99, top=top, bottom=bottom if h else 0.01, wspace=0.02, hspace=hspace)
    fig.savefig(os.path.join(a.salida, salida), dpi=dpi)
    plt.close(fig)


figura([("m1_color.png", "(a) Color real, tras el Módulo 1"), ("m1_clase.png", "(b) Clase real de cada punto")],
       1, 2, "captura_m1.png", CLASES)

figura([("m2_cabeza_SemanticPOSS.png", "(a) Cabeza predicha, HyperNCD original (SemanticPOSS)"),
        ("m2_cabeza_ArCH.png", "(b) Cabeza predicha, HyperNCD adaptado (ArCH)"),
        ("m2_novel_SemanticPOSS.png", "(c) Novedoso o conocido, original"),
        ("m2_novel_ArCH.png", "(d) Novedoso o conocido, adaptado")],
       2, 2, "captura_m2.png",
       [("novedoso", "#e07b39"), ("conocido", "#2b7bba")])

# M3: imagen mas chica en la pagina y letras mas grandes (se imprime a ~0.85 del ancho del texto, sin reducir la letra)
figura([("m3_pred_original.png", "(a) Clase predicha,\nSonata original"),
        ("m3_pred_adaptado.png", "(b) Clase predicha,\nSonata adaptado"),
        ("m3_acierto_original.png", "(c) Aciertos,\nSonata original"),
        ("m3_acierto_adaptado.png", "(d) Aciertos,\nSonata adaptado")],
       2, 2, "captura_m3.png",
       CLASES + [("sin clase", "#e6e6e6"), ("acierto", "#2ca02c"), ("error", "#d62728")],
       ancho=5.2, alto_panel=2.6, fuente=9, fuente_leyenda=8.5, dpi=220, ncol_leyenda=4, alto_leyenda=0.55,
       top=0.92, bottom=0.10, hspace=0.28)
