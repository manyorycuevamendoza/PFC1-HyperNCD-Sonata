"""Recorta las figuras de la tesis que usa el poster (solo los paneles, sin titulos ni leyendas).

Los rotulos se vuelven a escribir en LaTeX con letra grande; aqui solo se separan las vistas.
Uso (desde la raiz del repo):  python documentacion/poster/preparar_figuras.py
"""
from pathlib import Path
from PIL import Image

RAIZ = Path(__file__).resolve().parents[2]
IMG = RAIZ / "documentacion/tesis_latex/asesoria5/imagenes"
SAL = Path(__file__).resolve().parent / "fig"
SAL.mkdir(exist_ok=True)


def abrir(nombre):
    im = Image.open(IMG / nombre).convert("RGBA")
    fondo = Image.new("RGBA", im.size, "white")
    fondo.alpha_composite(im)
    return fondo.convert("RGB")


def recortar(im, caja, salida):
    im.crop(caja).save(SAL / salida, optimize=True)


# escena_m1_m4.png: vista lateral (x, z) de cada metodo (se omite el marco negro de cada panel)
esc = abrir("escena_m1_m4.png")
cajas_lateral = [(112, 430, 281, 747), (365, 430, 532, 747), (617, 430, 784, 747),
                 (869, 430, 1036, 747), (1121, 430, 1289, 747)]
nombres = ["real", "m2", "m3", "m4base", "m4"]
for n, (x0, y0, x1, y1) in zip(nombres, cajas_lateral):
    recortar(esc, (x0 + 4, y0 + 4, x1 - 3, y1 - 3), f"lateral_{n}.png")

# escena_etapas.png: M1 (b) y M3 (c), vista lateral
et = abrir("escena_etapas.png")
recortar(et, (527 + 4, 528 + 4, 747 - 3, 943 - 3), "m1_lateral.png")
recortar(et, (922 + 4, 528 + 4, 1142 - 3, 943 - 3), "m3_lateral.png")

# captura_m2.png: panel (b), grupos predichos por HyperNCD adaptado (ArCH)
c2 = abrir("captura_m2.png")
recortar(c2, (800, 75, 1335, 565), "m2_cabeza.png")

print("listo:", sorted(p.name for p in SAL.iterdir()))
