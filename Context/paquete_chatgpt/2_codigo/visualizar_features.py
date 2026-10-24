"""Visualiza y mide las features de Sonata (salida del Modulo 3).

Lee el .npz de extraer_features.py y la nube .npy del Modulo 1, y dibuja en
una sola imagen cuatro vistas de la misma escena:
    1. Color real (RGB).
    2. PCA de f_ssl a 3 componentes, usadas como RGB (como la Fig. 4 de Sonata):
       mismo color = features parecidas.
    3. Similitud coseno de cada punto con un punto de consulta (como el mapa
       del sofa de Sonata): rojo = parecido, azul = distinto.
    4. k-means sobre f_ssl: grupos que el encoder separa sin etiquetas.

Ademas imprime y guarda en .json medidas sin etiquetas: varianza explicada
por la PCA, rango efectivo de las features y tamano de cada grupo.

Uso:
    python experimentacion/modulo3_sonata/visualizar_features.py \
        --feats cache/PL13_feats.npz --data data/PL13_SCN0001.npy \
        --output experimentacion/resultados/modulo3_PL13 --html
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feats", required=True, help=".npz de extraer_features.py")
    ap.add_argument("--data", required=True, help=".npy del Modulo 1 (para el color)")
    ap.add_argument("--output", required=True, help="ruta destino sin extension")
    ap.add_argument("--puntos", type=int, default=150_000,
                    help="puntos a dibujar (submuestreo, default 150000)")
    ap.add_argument("--k", type=int, default=8, help="grupos de k-means (default 8)")
    ap.add_argument("--consulta", type=int, default=-1,
                    help="indice del punto de consulta (-1 = automatico)")
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--html", action="store_true",
                    help="genera ademas un .html 3D interactivo (requiere plotly)")
    ap.add_argument("--puntos-html", type=int, default=80_000,
                    help="puntos en la vista interactiva (default 80000)")
    return ap.parse_args()


def a_texto_rgb(c):
    c = (np.clip(c, 0, 1) * 255).astype(int)
    return [f"rgb({r},{g},{b})" for r, g, b in c]


def guardar_html(ruta, xyz, vistas, q_xyz, titulo):
    """Una sola escena 3D con botones para cambiar la coloracion."""
    import plotly.graph_objects as go
    nombres = list(vistas)
    fig = go.Figure([
        go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers",
                     marker=dict(size=1.5, color=vistas[nombres[0]]),
                     hoverinfo="skip", name="nube"),
        go.Scatter3d(x=[q_xyz[0]], y=[q_xyz[1]], z=[q_xyz[2]], mode="markers",
                     marker=dict(size=8, color="black", symbol="diamond"),
                     name="punto de consulta"),
    ])
    botones = [dict(label=n, method="restyle",
                    args=[{"marker.color": [vistas[n]]}, [0]]) for n in nombres]
    fig.update_layout(
        title=dict(text=titulo, y=0.98), showlegend=False, height=850,
        scene=dict(aspectmode="data", xaxis_visible=False,
                   yaxis_visible=False, zaxis_visible=False),
        updatemenus=[dict(type="buttons", direction="right", x=0.5, y=1.0,
                          xanchor="center", yanchor="bottom", buttons=botones)],
        margin=dict(l=0, r=0, t=110, b=0))
    fig.write_html(ruta, include_plotlyjs="cdn")


def normalizar(x):
    lo, hi = np.percentile(x, [2, 98], axis=0)
    return np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)


def kmeans(x, k, rng, iters=30):
    centros = x[rng.choice(len(x), k, replace=False)]
    for _ in range(iters):
        d = (x ** 2).sum(1)[:, None] - 2 * x @ centros.T + (centros ** 2).sum(1)[None]
        etiqueta = d.argmin(1)
        for j in range(k):
            if (etiqueta == j).any():
                centros[j] = x[etiqueta == j].mean(0)
    return centros


def main():
    args = parse_args()
    rng = np.random.default_rng(args.semilla)

    z = np.load(args.feats)
    feat = z["feat"].astype(np.float32)          # (M, 1088) en la rejilla de Sonata
    coord = z["coord"]                           # (M, 3)
    inverse = z["inverse"]                       # (N,) punto original -> rejilla
    datos = np.load(args.data)                   # (N, 9) X Y Z R G B Nx Ny Nz
    color = np.zeros((len(feat), 3), np.float32)
    color[inverse] = datos[:, 3:6] / 255.0       # color de un punto original por celda

    m = len(feat)
    print(f"features: {feat.shape}  (puntos originales: {len(inverse):,})")

    # --- PCA (ajustada sobre una muestra, aplicada a todos) ---
    muestra = rng.choice(m, min(m, 100_000), replace=False)
    media = feat[muestra].mean(0)
    _, s, vt = np.linalg.svd(feat[muestra] - media, full_matrices=False)
    var = s ** 2 / (s ** 2).sum()
    pca = (feat - media) @ vt[:3].T
    # rango efectivo: exp de la entropia del espectro (1 = colapsado, alto = rico)
    p = var[var > 0]
    rango_efectivo = float(np.exp(-(p * np.log(p)).sum()))

    # --- similitud coseno con un punto de consulta ---
    # Se resta la media: sin centrar, todas las features comparten una
    # componente comun y la similitud sale alta en toda la escena.
    fn = feat - media
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-8
    q = args.consulta if args.consulta >= 0 else int(rng.integers(m))
    sim = fn @ fn[q]

    # --- k-means sobre las features normalizadas ---
    centros = kmeans(fn[muestra], args.k, rng)
    grupo = np.empty(m, np.int64)
    for i in range(0, m, 200_000):
        bloque = fn[i:i + 200_000]
        grupo[i:i + 200_000] = (bloque @ centros.T
                                - 0.5 * (centros ** 2).sum(1)).argmax(1)
    tam = np.bincount(grupo, minlength=args.k)

    # --- figura ---
    vis = rng.choice(m, min(m, args.puntos), replace=False)
    xyz = coord[vis]
    paneles = [
        ("Color real (RGB)", color[vis], None),
        (f"PCA de $f_{{ssl}}$ ({100 * var[:3].sum():.0f}% var.)", normalizar(pca[vis]), None),
        ("Similitud con el punto ★", sim[vis], "coolwarm"),
        (f"k-means, k={args.k}", grupo[vis], "tab10"),
    ]
    fig = plt.figure(figsize=(16, 13))
    for i, (titulo, c, cmap) in enumerate(paneles, 1):
        ax = fig.add_subplot(2, 2, i, projection="3d")
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=c, cmap=cmap, s=0.3,
                   linewidths=0, vmin=-1 if cmap == "coolwarm" else None,
                   vmax=1 if cmap == "coolwarm" else None)
        if i == 3:
            ax.scatter(*coord[q], c="k", marker="*", s=300)
        ax.set_title(titulo, fontsize=14)
        ax.set_box_aspect(np.ptp(xyz, axis=0))
        ax.view_init(elev=35, azim=-60)
        ax.set_axis_off()
    fig.suptitle(Path(args.feats).stem, fontsize=16)
    fig.tight_layout()

    salida = Path(args.output)
    salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(salida.with_suffix(".png"), dpi=130, bbox_inches="tight")

    medidas = {
        "feats": args.feats,
        "n_rejilla": int(m),
        "dim": int(feat.shape[1]),
        "varianza_pca_3": round(float(var[:3].sum()), 4),
        "varianza_pca_10": round(float(var[:10].sum()), 4),
        "rango_efectivo": round(rango_efectivo, 1),
        "consulta": q,
        "kmeans_k": args.k,
        "kmeans_tamanos": tam.tolist(),
    }
    salida.with_suffix(".json").write_text(json.dumps(medidas, indent=2))

    if args.html:
        sub = rng.choice(m, min(m, args.puntos_html), replace=False)
        paleta = plt.get_cmap("tab10")(np.arange(10))[:, :3]
        sim_col = plt.get_cmap("coolwarm")((sim[sub] + 1) / 2)[:, :3]
        vistas = {
            "Color real": a_texto_rgb(color[sub]),
            "PCA de f_ssl": a_texto_rgb(normalizar(pca[sub])),
            "Similitud con ◆": a_texto_rgb(sim_col),
            f"k-means (k={args.k})": a_texto_rgb(paleta[grupo[sub] % 10]),
        }
        guardar_html(salida.with_suffix(".html"), coord[sub], vistas, coord[q],
                     f"{Path(args.feats).stem}: gira con el mouse, cambia la vista con los botones")
        print(f"[ok] {salida.with_suffix('.html')}")
    for clave, valor in medidas.items():
        print(f"{clave:16s} {valor}")
    print(f"[ok] {salida.with_suffix('.png')}")


if __name__ == "__main__":
    main()
