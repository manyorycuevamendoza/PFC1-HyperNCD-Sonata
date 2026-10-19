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
    python visualizar_features.py --feats cache/PL13_feats.npz \
        --data data/PL13_SCN0001.npy --output figuras/PL13
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
    return ap.parse_args()


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
    for clave, valor in medidas.items():
        print(f"{clave:16s} {valor}")
    print(f"[ok] {salida.with_suffix('.png')}")


if __name__ == "__main__":
    main()
