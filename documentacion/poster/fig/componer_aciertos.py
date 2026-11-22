"""Mapa de aciertos y errores de M4: compara, pixel a pixel, la vista con etiqueta real y la vista predicha.
Aproximacion visual (no por punto): un pixel es acierto si su color coincide con el real en un entorno de 3x3."""
from pathlib import Path
from PIL import Image
import numpy as np

AQUI = Path(__file__).resolve().parent
ACIERTO, ERROR, S = (46, 125, 210), (220, 38, 38), 4

def mapa(v):
    r = np.array(Image.open(AQUI / f"{v}_real.png").convert("RGB").crop((0, 0, 161, 310))).astype(int)
    m = np.array(Image.open(AQUI / f"{v}_m4.png").convert("RGB")).astype(int)
    pm = m.sum(2) < 740
    d = np.full(m.shape[:2], 1e9)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            d = np.minimum(d, np.abs(np.roll(np.roll(r, dy, 0), dx, 1) - m).sum(2))
    ok = pm & (d < 60)
    out = np.full(m.shape, 255, np.uint8)
    out[ok] = ACIERTO; out[pm & ~ok] = ERROR
    out[:, :3] = 255; out[:, -3:] = 255
    print(v, "aciertos %.1f %%" % (100 * ok.sum() / pm.sum()))
    return Image.fromarray(out).resize((161 * S, 310 * S), Image.NEAREST)

a, b = mapa("planta"), mapa("lateral")
o = Image.new("RGB", (a.width + b.width + 60, a.height), "white")
o.paste(a, (0, 0)); o.paste(b, (a.width + 60, 0))
o.save(AQUI / "aciertos_errores.png", optimize=True)
a.save(AQUI / "aciertos_planta.png", optimize=True)
