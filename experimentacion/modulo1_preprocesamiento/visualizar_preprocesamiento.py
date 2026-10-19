"""Vista 3D interactiva del Modulo 1: el escaneo antes y despues de preprocesar.

Genera un .html con una escena 3D y botones para alternar entre:
    - Crudo (color real): muestra del .las original, centrada igual que la salida.
    - Preprocesado (color real): la matriz S que sale de preprocesar.py.
    - Preprocesado (normales): la normal de cada punto usada como color.
Abajo muestra los conteos de cada paso, leidos del .json de metadatos.

Uso:
    python experimentacion/modulo1_preprocesamiento/visualizar_preprocesamiento.py \
        --las  experimentacion/datos/modulo1_PL7/PL7_SCN0001.las \
        --npy  experimentacion/datos/modulo1_PL7/PL7_SCN0001_preprocesado.npy \
        --json experimentacion/datos/modulo1_PL7/PL7_SCN0001_preprocesado.json \
        --output experimentacion/resultados/modulo1_PL7

Si no se tiene el .las, se omite --las y se muestra solo el preprocesado.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--las", help=".las original (opcional)")
    ap.add_argument("--npy", required=True, help="salida .npy de preprocesar.py")
    ap.add_argument("--json", help=".json de metadatos (opcional, da los conteos)")
    ap.add_argument("--output", required=True, help="ruta destino sin extension")
    ap.add_argument("--puntos", type=int, default=100_000,
                    help="puntos por vista (submuestreo, default 100000)")
    ap.add_argument("--semilla", type=int, default=0)
    return ap.parse_args()


def a_texto_rgb(c):
    c = (np.clip(c, 0, 1) * 255).astype(int)
    return [f"rgb({r},{g},{b})" for r, g, b in c]


def traza(xyz, colores, nombre, visible):
    return go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers",
                        marker=dict(size=1.5, color=colores), hoverinfo="skip",
                        name=nombre, visible=visible)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.semilla)
    meta = json.load(open(args.json)) if args.json else {}

    S = np.load(args.npy)
    sub = rng.choice(len(S), min(len(S), args.puntos), replace=False)
    vistas = [
        ("Preprocesado: color real", S[sub, :3], a_texto_rgb(S[sub, 3:6] / 255)),
        ("Preprocesado: normales", S[sub, :3], a_texto_rgb((S[sub, 6:9] + 1) / 2)),
    ]

    if args.las:
        import laspy
        las = laspy.read(args.las)
        i = rng.choice(len(las.x), min(len(las.x), args.puntos), replace=False)
        xyz = np.stack([las.x[i], las.y[i], las.z[i]], 1)
        # mismo centrado que preprocesar.py, para que ambas vistas coincidan
        offset = np.array(meta["offset_utm"]) if "offset_utm" in meta else \
            (xyz.min(0) + xyz.max(0)) / 2
        rgb = np.stack([las.red[i], las.green[i], las.blue[i]], 1).astype(np.float32)
        rgb /= 65535.0 if rgb.max() > 255 else 255.0
        vistas.insert(0, ("Crudo: color real", xyz - offset, a_texto_rgb(rgb)))

    fig = go.Figure([traza(x, c, n, k == 0) for k, (n, x, c) in enumerate(vistas)])
    botones = [dict(label=n, method="update",
                    args=[{"visible": [j == k for j in range(len(vistas))]}])
               for k, (n, _, _) in enumerate(vistas)]

    nombre = Path(args.npy).stem.replace("_preprocesado", "")
    conteos = ""
    if meta:
        conteos = (f"Crudo {meta['n_crudo']:,} → SOR {meta['n_sor']:,} → "
                   f"vóxel {meta['voxel'] * 100:g} cm {meta['n_voxel']:,} puntos  "
                   f"(q={meta['q']}, λ={meta['lambda']:g}). "
                   f"Cada vista muestra {args.puntos:,} puntos al azar.")
    fig.update_layout(
        title=dict(text=f"Módulo 1, {nombre}: gira con el mouse, cambia la vista "
                        f"con los botones", y=0.98),
        showlegend=False, height=850,
        scene=dict(aspectmode="data", xaxis_visible=False,
                   yaxis_visible=False, zaxis_visible=False),
        updatemenus=[dict(type="buttons", direction="right", x=0.5, y=1.0,
                          xanchor="center", yanchor="bottom", buttons=botones)],
        annotations=[dict(text=conteos, showarrow=False, x=0.5, y=0,
                          xref="paper", yref="paper", font=dict(size=13))],
        margin=dict(l=0, r=0, t=110, b=30))

    salida = Path(args.output).with_suffix(".html")
    salida.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(salida, include_plotlyjs="cdn")
    print(f"[ok] {salida}")


if __name__ == "__main__":
    main()
