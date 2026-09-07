"""Extrae features SSL de Sonata (encoder congelado) y las cachea a disco.

Es el Paso 1 del pipeline del OE2: Sonata corre una sola vez por seccion,
guarda f_ssl en disco, y HyperNCD lo consume despues desde su propio entorno
conda (las dos pilas de CUDA son incompatibles, por eso el disco de por medio).

Entrada : .txt ASCII de 9 columnas -> X Y Z R G B Nx Ny Nz
Salida  : .npz con feat (float16), coord, inverse y metadatos

Uso:
    python extraer_features.py --input data/Otros.txt \
                               --output cache/Otros_feats.npz
"""

import argparse
import time

import numpy as np
import torch

import sonata


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="nube de puntos .txt (9 columnas)")
    ap.add_argument("--output", required=True, help="destino .npz")
    ap.add_argument("--grid", type=float, default=0.02,
                    help="tamano de rejilla en metros (default: 0.02)")
    ap.add_argument("--fp32", action="store_true",
                    help="guardar en float32 en vez de float16")
    return ap.parse_args()


def cargar_nube(ruta):
    """Lee el .txt y lo arma en el dict que espera Sonata."""
    datos = np.loadtxt(ruta, dtype=np.float32)
    if datos.ndim != 2 or datos.shape[1] != 9:
        raise ValueError(
            f"{ruta}: se esperaban 9 columnas (XYZ RGB NxNyNz), "
            f"se encontraron {datos.shape}"
        )
    return {
        "coord": np.ascontiguousarray(datos[:, 0:3]),
        "color": np.ascontiguousarray(datos[:, 3:6]),
        "normal": np.ascontiguousarray(datos[:, 6:9]),
    }


def cargar_modelo():
    """Sonata preentrenado, sin flash-attn (no esta instalado en Khipu)."""
    config = dict(enc_patch_size=[1024 for _ in range(5)], enable_flash=False)
    modelo = sonata.load("sonata", repo_id="facebook/sonata",
                         custom_config=config).cuda()
    modelo.eval()
    return modelo


def upcast(point):
    """Devuelve las features a la escala de la rejilla.

    Sonata es un encoder jerarquico: la salida esta submuestreada varias veces.
    Este bloque deshace ese pooling. Los dos primeros niveles se concatenan
    (conservan el detalle fino) y el resto se propaga por indice. Es el
    procedimiento documentado en el README de Sonata.
    """
    for _ in range(2):
        assert "pooling_parent" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
        point = parent
    while "pooling_parent" in point.keys():
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = point.feat[inverse]
        point = parent
    return point


def main():
    args = parse_args()

    print(f"[1/4] leyendo {args.input}")
    t0 = time.time()
    point = cargar_nube(args.input)
    n_original = point["coord"].shape[0]
    print(f"      {n_original} puntos ({time.time() - t0:.1f}s)")

    print("[2/4] cargando Sonata preentrenado")
    modelo = cargar_modelo()

    print(f"[3/4] transform (GridSample = {args.grid} m)")
    point = sonata.transform.default()(point)
    n_rejilla = point["coord"].shape[0]
    print(f"      {n_original} -> {n_rejilla} puntos "
          f"({100 * n_rejilla / n_original:.1f}% retenido)")

    print("[4/4] forward")
    t0 = time.time()
    with torch.inference_mode():
        for clave in list(point.keys()):
            if isinstance(point[clave], torch.Tensor):
                point[clave] = point[clave].cuda(non_blocking=True)
        point = modelo(point)
        point = upcast(point)

    dtype = np.float32 if args.fp32 else np.float16
    feat = point.feat.cpu().numpy().astype(dtype)
    coord = point.coord.cpu().numpy().astype(np.float32)
    # inverse mapea de la rejilla a los puntos originales: feat[inverse]
    inverse = point.inverse.cpu().numpy().astype(np.int32)

    print(f"      forward en {time.time() - t0:.1f}s")
    print(f"\n>>> FEATURES SSL: {feat.shape}   dim f_ssl = {feat.shape[1]}")

    np.savez_compressed(
        args.output,
        feat=feat,
        coord=coord,
        inverse=inverse,
        grid_size=np.float32(args.grid),
        n_original=np.int64(n_original),
    )
    print(f"guardado -> {args.output}")


if __name__ == "__main__":
    main()
