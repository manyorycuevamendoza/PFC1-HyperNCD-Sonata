"""Flujos ilustrados de los Modulos 1 a 4 (capitulo 3), con las herramientas de cada paso.

Cada caja es un paso del modulo y debajo lleva el logo de la herramienta que lo hace. Los logos de NumPy, Open3D y
PyTorch vienen de imagenes/logos. Las demas herramientas salen como etiqueta de texto, porque no hay archivo de logo.

Uso (desde la raiz del repo):
    python experimentacion/escena_capilla_etiquetada/figura_modulos.py --imagenes documentacion/tesis_latex/asesoria5/imagenes
"""
import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ap = argparse.ArgumentParser()
ap.add_argument("--imagenes", required=True)
a = ap.parse_args()
IM = a.imagenes
LOGOS = {n: imread(os.path.join(IM, "logos", f"{n}.png")) for n in ("numpy", "open3d", "pytorch")}

COLOR = {"dato": "#fde6cf", "paso": "#dbe5fb", "modelo": "#e8dcf7"}


def herramienta(ax, cx, y, nombre):
    """Logo si existe, y etiqueta de texto si no."""
    if nombre in LOGOS:
        im = LOGOS[nombre]
        h = 0.34
        w = h * im.shape[1] / im.shape[0]
        w = min(w, 0.9)
        h = w * im.shape[0] / im.shape[1]
        ax.imshow(im, extent=(cx - w / 2, cx + w / 2, y, y + h), zorder=4, aspect="auto")
    else:
        ax.text(cx, y + 0.17, nombre, ha="center", va="center", fontsize=6.6, color="#202030", zorder=5,
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#8a8aa0", lw=0.7))


def tira(nombre_archivo, titulo, pasos):
    """pasos = [(tipo, texto, [herramientas])]"""
    n = len(pasos)
    ancho = 9.4
    paso_x = ancho / n
    fig, ax = plt.subplots(figsize=(ancho, 1.62), dpi=190)
    ax.set_xlim(0, ancho)
    ax.set_ylim(0, 1.62)
    ax.axis("off")
    ax.text(0.05, 1.52, titulo, fontsize=8.6, fontweight="bold", va="center", color="#202030")
    for i, (tipo, texto, tools) in enumerate(pasos):
        x = i * paso_x + 0.06
        w = paso_x - 0.2
        ax.add_patch(FancyBboxPatch((x, 0.5), w, 0.9, boxstyle="round,pad=0.015,rounding_size=0.07",
                                    fc=COLOR[tipo], ec="#44445a", lw=1.0))
        ax.text(x + w / 2, 0.95, texto, ha="center", va="center", fontsize=7.0, color="#202030", linespacing=1.12)
        for k, t in enumerate(tools):
            herramienta(ax, x + w * (k + 1) / (len(tools) + 1), 0.07, t)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.01, 0.95), (x + w + 0.17, 0.95), arrowstyle="-|>", mutation_scale=11,
                                         lw=1.6, color="#33334a"))
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(os.path.join(IM, nombre_archivo))
    plt.close(fig)


tira("flujo_m1.png", "Módulo 1. Preprocesamiento", [
    ("dato", "Archivo LAS\nde un escaneo", ["laspy"]),
    ("paso", "1. Carga por\nbloques", ["laspy", "numpy"]),
    ("paso", "2. Centrado\nde coordenadas", ["numpy"]),
    ("paso", "3. Filtrado SOR\n(puntos sueltos)", ["open3d"]),
    ("paso", "4. Voxelización\nde 5 cm", ["open3d"]),
    ("paso", "5. Normales\ncon 20 vecinos", ["open3d"]),
    ("dato", "6. Matriz S\n(9 columnas)", ["numpy"]),
])
tira("flujo_m2.png", "Módulo 2. HyperNCD", [
    ("dato", "ArCH o\nSemanticPOSS", ["numpy"]),
    ("paso", "Bloques de 10 m,\n60 000 puntos", ["numpy"]),
    ("modelo", "Entrenamiento\n30 épocas", ["MinkowskiEngine", "pytorch"]),
    ("paso", "Pseudo-etiquetas\nSinkhorn-Knopp", ["pytorch"]),
    ("modelo", "Mejor modelo\npor validación", ["pytorch"]),
    ("dato", "f_sem de la\nescena (.npz)", ["numpy"]),
])
tira("flujo_m3.png", "Módulo 3. Sonata", [
    ("dato", "Matriz S\nde la sección", ["numpy"]),
    ("paso", "Pesos original\no adaptado", ["pytorch", "Pointcept"]),
    ("paso", "Rejilla de 2 cm\ne índice inverso", ["numpy"]),
    ("modelo", "Codificador PTv3\n(5 etapas)", ["pytorch"]),
    ("paso", "Mediciones de\ncontrol", ["numpy"]),
    ("dato", "f_ssl de 1088\nvalores (.npz)", ["numpy"]),
])
tira("flujo_m4.png", "Módulo 4. Fusión y agrupamiento", [
    ("dato", "f_sem, f_ssl y\nmatriz S", ["numpy"]),
    ("paso", "Alinear con el\níndice inverso", ["numpy"]),
    ("paso", "Regiones de\n0,75 m", ["numpy", "SciPy"]),
    ("paso", "Afinidad e\nhiperaristas", ["numpy"]),
    ("modelo", "Prototipos Sinkhorn-\nKnopp, 15 iteraciones", ["numpy"]),
    ("dato", "Grupo de\ncada punto", ["numpy"]),
])
print("listo")
