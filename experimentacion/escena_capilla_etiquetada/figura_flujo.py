"""Figuras ilustradas del flujo general y de la dependencia entre modulos (capitulo 3).

Cada etapa lleva una miniatura de lo que produce, tomada de las figuras de la tesis sobre la capilla etiquetada,
y un icono que dice de que tipo es (geometria, red neuronal, hipergrafo, medida). No calcula nada nuevo.

Uso (desde la raiz del repo):
    python experimentacion/escena_capilla_etiquetada/figura_flujo.py --imagenes documentacion/tesis_latex/asesoria5/imagenes
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon

ap = argparse.ArgumentParser()
ap.add_argument("--imagenes", required=True)
a = ap.parse_args()
IM = a.imagenes


def leer(nombre):
    return imread(os.path.join(IM, nombre))[..., :3]


def autocorte(im, margen=6):
    m = (im < 0.96).any(axis=2)
    ys, xs = np.where(m)
    return im[max(ys.min() - margen, 0):ys.max() + margen, max(xs.min() - margen, 0):xs.max() + margen]


def trozo(im, x0, x1, y0, y1):
    h, w = im.shape[:2]
    c = im[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
    c = c[4:-4, 4:-4]                      # quita el marco del eje
    return autocorte(c)


etapas = leer("escena_etapas.png")
cuatro = leer("escena_m1_m4.png")
m2 = leer("captura_m2.png")
m3 = leer("captura_m3.png")

mini = {
    "crudo": trozo(etapas, 0.115, 0.285, 0.115, 0.50),
    "m1": trozo(etapas, 0.34, 0.67, 0.54, 1.0),
    "pca": trozo(etapas, 0.67, 1.0, 0.54, 1.0),
    "m2": autocorte(m2[int(0.08 * m2.shape[0]):int(0.465 * m2.shape[0]), int(0.5 * m2.shape[1]):]),
    "m3": autocorte(m3[int(0.08 * m3.shape[0]):int(0.465 * m3.shape[0]), :int(0.5 * m3.shape[1])]),
    "m4": trozo(cuatro, 0.8, 1.0, 0.52, 0.95),
}

NARANJA, AZUL, MORADO = "#fde6cf", "#dbe5fb", "#e8dcf7"


def caja(ax, x, y, w, h, color, titulo, texto, miniatura=None, icono=None, sub=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=color, ec="#44445a", lw=1.2))
    ax.text(x + w / 2, y + h - 0.18, titulo, ha="center", va="top", fontsize=9.6, fontweight="bold", color="#202030")
    if miniatura is not None:
        ih, iw = miniatura.shape[:2]
        alto = h - 1.15
        ancho = alto * iw / ih
        if ancho > w - 0.5:
            ancho = w - 0.5
            alto = ancho * ih / iw
        cx, cy = x + w / 2, y + 0.62 + (h - 1.15) / 2
        ax.imshow(miniatura, extent=(cx - ancho / 2, cx + ancho / 2, cy - alto / 2, cy + alto / 2), zorder=3)
    ax.text(x + w / 2, y + 0.1, texto, ha="center", va="bottom", fontsize=7.4, color="#303040", linespacing=1.12)
    if icono is not None:
        icono(ax, x + w - 0.5, y + h / 2 + 0.15)


def flecha(ax, p, q, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=16, lw=2.0, color="#33334a",
                                 connectionstyle=f"arc3,rad={rad}", zorder=5))


def icono_geometria(ax, cx, cy, s=0.3):
    d = s
    top = [(cx, cy + d), (cx + d, cy + d / 2), (cx, cy), (cx - d, cy + d / 2)]
    izq = [(cx - d, cy + d / 2), (cx, cy), (cx, cy - d), (cx - d, cy - d / 2)]
    der = [(cx + d, cy + d / 2), (cx, cy), (cx, cy - d), (cx + d, cy - d / 2)]
    for pts, c in ((top, "#ffffff"), (izq, "#c9d3ee"), (der, "#9fb0e0")):
        ax.add_patch(Polygon(pts, closed=True, fc=c, ec="#33334a", lw=1.0, zorder=6))


def icono_red(ax, cx, cy, s=0.3, color="#6b4fb0"):
    capas = [(-s, [0.5, 0, -0.5]), (0, [0.75, 0.25, -0.25, -0.75]), (s, [0.5, -0.5])]
    pts = [[(cx + dx, cy + dy * s) for dy in ys] for dx, ys in capas]
    for i in range(len(pts) - 1):
        for p in pts[i]:
            for q in pts[i + 1]:
                ax.plot([p[0], q[0]], [p[1], q[1]], color="#8a8aa0", lw=0.6, zorder=6)
    for capa in pts:
        for p in capa:
            ax.add_patch(Circle(p, 0.045, fc=color, ec="#33334a", lw=0.6, zorder=7))


def icono_hiper(ax, cx, cy, s=0.3):
    ax.add_patch(Ellipse((cx, cy), 2.0 * s, 1.3 * s, fc="#f2c94c", alpha=0.45, ec="#33334a", lw=0.9, zorder=6))
    for dx, dy in ((-0.55, -0.15), (0.55, -0.15), (0.0, 0.3)):
        ax.add_patch(Circle((cx + dx * s, cy + dy * s), 0.05, fc="#33334a", zorder=7))
    ax.plot([cx - 0.55 * s, cx + 0.55 * s, cx, cx - 0.55 * s], [cy - 0.15 * s, cy - 0.15 * s, cy + 0.3 * s, cy - 0.15 * s],
            color="#33334a", lw=0.8, zorder=7)


def icono_barras(ax, cx, cy, s=0.3):
    for i, hh in enumerate((0.5, 0.9, 0.7)):
        ax.add_patch(plt.Rectangle((cx - s + i * 0.38 * s * 1.6, cy - s * 0.8), 0.3 * s * 1.4, hh * s * 1.5, fc="#4c9a6a", ec="#33334a", lw=0.8, zorder=6))


def icono_filtro(ax, cx, cy, s=0.3):
    ax.add_patch(Polygon([(cx - s, cy + s * 0.8), (cx + s, cy + s * 0.8), (cx + 0.15 * s, cy - 0.1 * s), (cx + 0.15 * s, cy - s * 0.9),
                          (cx - 0.15 * s, cy - s * 0.7), (cx - 0.15 * s, cy - 0.1 * s)], closed=True, fc="#ffffff", ec="#33334a", lw=1.0, zorder=6))


def grafico_barras():
    fig, ax = plt.subplots(figsize=(3.2, 2.2), dpi=150)
    nombres = ["M2", "M3", "M4\nbase", "M4"]
    medias = [48.4, 45.4, 51.4, 52.5]
    desv = [1.1, 8.5, 1.7, 2.5]
    ax.bar(nombres, medias, yerr=desv, color=["#5d8fd6", "#9a6fd0", "#e0a24a", "#4c9a6a"], capsize=3, ec="#33334a", lw=0.8)
    ax.set_ylim(0, 70)
    ax.set_ylabel("mIoU (%)", fontsize=8)
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3] / 255.0
    plt.close(fig)
    return autocorte(img, 4)


barras = grafico_barras()

# ---------------- figura 1, flujo general ----------------
fig, ax = plt.subplots(figsize=(9.4, 5.0), dpi=190)
ax.set_xlim(0, 12.4)
ax.set_ylim(0, 6.6)
ax.axis("off")
W, H = 3.55, 2.9
x1, x2, x3 = 0.2, 4.4, 8.6
yA, yB = 3.55, 0.15
caja(ax, x1, yA, W, H, NARANJA, "E0. Datos", "23 escaneos LAS de la Catedral\n(589 millones de puntos), sin etiquetas", mini["crudo"])
caja(ax, x2, yA, W, H, AZUL, "E1. Preparación (M1)", "Se limpia el escaneo y se convierte\nen la matriz S que usan M2 y M3", mini["m1"], icono_filtro)
caja(ax, x3, yA, W, H, MORADO, "E2. HyperNCD (M2)", "Aprende con etiquetas de otro conjunto\ny entrega un descriptor semántico", mini["m2"], icono_red)
caja(ax, x3, yB, W, H, MORADO, "E3. Sonata (M3)", "Aprende sin etiquetas, original y adaptado,\ny entrega un descriptor auto-supervisado", mini["pca"], lambda ax_, x, y: icono_red(ax_, x, y, color="#2e8b74"))
caja(ax, x2, yB, W, H, AZUL, "E4. Fusión (M4)", "Se unen los dos descriptores\ny se agrupan los puntos en 8 grupos", mini["m4"], icono_hiper)
caja(ax, x1, yB, W, H, NARANJA, "E5. Evaluación", "mIoU, IoU por clase, NMI y ARI\nfrente a la etiqueta real", barras)
flecha(ax, (x1 + W + 0.04, yA + H / 2), (x2 - 0.04, yA + H / 2))
flecha(ax, (x2 + W + 0.04, yA + H / 2), (x3 - 0.04, yA + H / 2))
flecha(ax, (x3 + W / 2, yA - 0.04), (x3 + W / 2, yB + H + 0.04))
flecha(ax, (x3 - 0.04, yB + H / 2), (x2 + W + 0.04, yB + H / 2))
flecha(ax, (x2 - 0.04, yB + H / 2), (x1 + W + 0.04, yB + H / 2))
fig.subplots_adjust(0, 0, 1, 1)
fig.savefig(os.path.join(IM, "flujo_general.png"))
plt.close(fig)

# ---------------- figura 2, dependencia entre modulos ----------------
fig, ax = plt.subplots(figsize=(9.4, 4.25), dpi=190)
ax.set_xlim(0, 12.4)
ax.set_ylim(0, 5.6)
ax.axis("off")
w, h = 2.75, 2.45
caja(ax, 0.15, 1.58, w, h, NARANJA, "M1. Preproceso", "Matriz S de cada escaneo", mini["m1"], icono_filtro)
caja(ax, 3.55, 3.0, w, h - 0.1, AZUL, "M2. HyperNCD", "Descriptor semántico", mini["m2"], icono_red)
caja(ax, 3.55, 0.1, w, h - 0.1, AZUL, "M3. Sonata", "Descriptor auto-supervisado", mini["pca"], lambda ax_, x, y: icono_red(ax_, x, y, color="#2e8b74"))
caja(ax, 6.7, 1.58, w, h, MORADO, "M4. Fusión", "Une los descriptores y agrupa", mini["m4"], icono_hiper)
caja(ax, 9.9, 1.58, 2.35, h, NARANJA, "Evaluación", "mIoU y otras métricas", barras)
flecha(ax, (0.15 + w + 0.04, 3.0), (3.5, 4.1), 0.1)
flecha(ax, (0.15 + w + 0.04, 2.6), (3.5, 1.3), -0.1)
flecha(ax, (3.55 + w + 0.04, 4.1), (6.65, 3.4), -0.1)
flecha(ax, (3.55 + w + 0.04, 1.3), (6.65, 2.2), 0.1)
flecha(ax, (6.7 + w + 0.04, 2.8), (9.85, 2.8))
ax.text(4.9, 2.62, "en paralelo", ha="center", va="center", fontsize=8.4, style="italic", color="#33334a",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#9a9ab0"))
fig.subplots_adjust(0, 0, 1, 1)
fig.savefig(os.path.join(IM, "flujo_dependencias.png"))
plt.close(fig)
print("listo")
