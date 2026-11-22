"""Compone la imagen inicial (nube cruda RGB) y la final (resultado M4) con planta y vista lateral."""
from pathlib import Path
from PIL import Image

AQUI = Path(__file__).resolve().parent
IMG = AQUI.parents[1] / "tesis_latex/asesoria5/imagenes"

def abrir(p):
    im = Image.open(p).convert("RGBA")
    f = Image.new("RGBA", im.size, "white"); f.alpha_composite(im)
    return f.convert("RGB")

def unir(a, b, sep=24):
    h = max(a.height, b.height)
    out = Image.new("RGB", (a.width + sep + b.width, h), "white")
    out.paste(a, (0, (h - a.height) // 2)); out.paste(b, (a.width + sep, (h - b.height) // 2))
    return out

et = abrir(IMG / "escena_etapas.png")
planta = et.crop((138 + 4, 96 + 4, 345 - 3, 512 - 3))
lateral = et.crop((132 + 4, 528 + 4, 350 - 3, 943 - 3))
unir(planta, lateral).save(AQUI / "inicial.png", optimize=True)
def ajustar(im, h):
    return im.resize((round(im.width * h / im.height), h), Image.LANCZOS)

pm = abrir(AQUI / "planta_m4.png"); pm = pm.crop((4, 4, pm.width - 4, pm.height - 4))
lm = abrir(AQUI / "lateral_m4.png"); lm = lm.crop((4, 4, lm.width - 4, lm.height - 4))
unir(ajustar(pm, planta.height), ajustar(lm, lateral.height)).save(AQUI / "final.png", optimize=True)
