"""Une varias capturas del visor 3D en una lamina horizontal para el informe.

Las capturas se exportan desde visor_3d.py con el boton de la camara (o en
lote, ver experimentacion/README.md) y comparten angulo y escala. Este script
recorta el blanco sobrante, las pone lado a lado y escribe un titulo sobre cada
una.

Uso (desde la raiz del repo):
    python experimentacion/figuras_informe/componer_lamina.py \
        --imagenes experimentacion/resultados/informe/PL13_0_Crudo.png \
                   experimentacion/resultados/informe/PL13_1_M1_color_real.png \
                   experimentacion/resultados/informe/PL13_3_M3_PCA_de_f_ssl.png \
        --titulos "(a) Entrada" "(b) Módulo 1" "(c) Módulo 3" \
        --output experimentacion/resultados/informe/lamina_PL13.png
"""

import argparse

from PIL import Image, ImageChops, ImageDraw, ImageFont


def recortar(img, margen=20):
    fondo = Image.new(img.mode, img.size, (255, 255, 255))
    caja = ImageChops.difference(img, fondo).getbbox()
    if not caja:
        return img
    x0, y0, x1, y1 = caja
    return img.crop((max(0, x0 - margen), max(0, y0 - margen),
                     min(img.width, x1 + margen), min(img.height, y1 + margen)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--imagenes", nargs="+", required=True)
    ap.add_argument("--titulos", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if len(args.imagenes) != len(args.titulos):
        raise SystemExit("debe haber un titulo por imagen")

    imgs = [Image.open(p).convert("RGB") for p in args.imagenes]
    # misma caja de recorte para todas, para no romper la escala comun
    cajas = [ImageChops.difference(i, Image.new("RGB", i.size, "white")).getbbox()
             for i in imgs]
    x0 = min(c[0] for c in cajas) - 20; y0 = min(c[1] for c in cajas) - 20
    x1 = max(c[2] for c in cajas) + 20; y1 = max(c[3] for c in cajas) + 20
    imgs = [i.crop((max(0, x0), max(0, y0), min(i.width, x1), min(i.height, y1)))
            for i in imgs]

    alto_titulo = 70
    w, h = imgs[0].size
    lamina = Image.new("RGB", (w * len(imgs), h + alto_titulo), "white")
    dibujo = ImageDraw.Draw(lamina)
    try:
        fuente = ImageFont.truetype("arial.ttf", 40)
    except OSError:
        fuente = ImageFont.load_default()
    for k, (img, titulo) in enumerate(zip(imgs, args.titulos)):
        lamina.paste(img, (k * w, alto_titulo))
        ancho = dibujo.textlength(titulo, font=fuente)
        dibujo.text((k * w + (w - ancho) / 2, 12), titulo, fill="black", font=fuente)
    lamina.save(args.output)
    print(f"[ok] {args.output}  {lamina.size}")


if __name__ == "__main__":
    main()
