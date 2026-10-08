"""Modulo 1 del pipeline: preprocesamiento de un levantamiento LAS.

Lee una o varias nubes .las, las une en una sola seccion S_j y entrega la
nube limpia y voxelizada que consume extraer_features.py (Modulo 3).

Pasos (capitulo 2, Modulo 1):
    1. Lectura por bloques: XYZ + RGB. Las coordenadas se centran restando
       el centro de la seccion (vienen en UTM, ~10^6 m).
    2. Limpieza de ruido por SOR con q vecinos y umbral lambda.
    3. Voxelizacion: promedio de XYZ y RGB por voxel.
    4. Normales por PCA de los mismos q vecinos.

Entrada : uno o varios .las (mismo sistema de coordenadas)
Salida  : <salida>.npy  float32 (N, 9) -> X Y Z R G B Nx Ny Nz
          <salida>.json metadatos (offset para volver a UTM, conteos)

Uso:
    python preprocesar.py --input datos_crudos/.../PL4_SCN0001.las \
                          --output data/PL4
"""

import argparse
import json
import time
from pathlib import Path

import laspy
import numpy as np
from scipy.spatial import cKDTree


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, nargs="+",
                    help="uno o varios .las que forman la seccion")
    ap.add_argument("--output", required=True,
                    help="ruta destino sin extension (se crean .npy y .json)")
    ap.add_argument("--voxel", type=float, default=0.05,
                    help="tamano de voxel en metros (default: 0.05)")
    ap.add_argument("--q", type=int, default=20,
                    help="vecinos para SOR y normales (default: 20)")
    ap.add_argument("--lam", type=float, default=2.0,
                    help="umbral lambda del SOR (default: 2.0)")
    ap.add_argument("--chunk", type=int, default=5_000_000,
                    help="puntos por bloque al leer y consultar el KD-tree")
    ap.add_argument("--prevoxel", type=float, default=0.0,
                    help="voxel previo al SOR en metros para nubes muy densas "
                         "(0 = desactivado, sigue el orden de la tesis)")
    return ap.parse_args()


def leer_las(rutas, chunk):
    """Lee los .las por bloques y devuelve XYZ (float64) y RGB (0-255)."""
    xyz, rgb = [], []
    for ruta in rutas:
        with laspy.open(ruta) as f:
            for bloque in f.chunk_iterator(chunk):
                xyz.append(np.stack([bloque.x, bloque.y, bloque.z], axis=1))
                rgb.append(np.stack([bloque.red, bloque.green, bloque.blue],
                                    axis=1).astype(np.float32))
    xyz = np.concatenate(xyz)
    rgb = np.concatenate(rgb)
    # LAS guarda el color en 16 bits; Sonata lo espera en 0-255.
    if rgb.max() > 255:
        rgb /= 256.0
    return xyz, rgb


def voxelizar(xyz, rgb, lado):
    """Un representante por voxel: promedio de posicion y color."""
    celda = np.floor(xyz / lado).astype(np.int64)
    celda -= celda.min(axis=0)
    dim = celda.max(axis=0) + 1
    # Indice lineal de la celda: unique sobre 1 columna es mucho mas rapido.
    clave = (celda[:, 0] * dim[1] + celda[:, 1]) * dim[2] + celda[:, 2]
    _, inv, cuenta = np.unique(clave, return_inverse=True, return_counts=True)
    xyz_v = np.stack([np.bincount(inv, xyz[:, c]) for c in range(3)], 1)
    rgb_v = np.stack([np.bincount(inv, rgb[:, c]) for c in range(3)], 1)
    xyz_v /= cuenta[:, None]
    rgb_v /= cuenta[:, None]
    return xyz_v, rgb_v.astype(np.float32)


def sor(xyz, q, lam, chunk):
    """Statistical Outlier Removal: descarta puntos cuya distancia media a
    sus q vecinos supera mu + lambda * sigma."""
    arbol = cKDTree(xyz)
    media = np.empty(len(xyz), np.float32)
    for i in range(0, len(xyz), chunk):
        d, _ = arbol.query(xyz[i:i + chunk], k=q + 1, workers=-1)
        media[i:i + chunk] = d[:, 1:].mean(axis=1)  # [:,0] es el propio punto
    umbral = media.mean() + lam * media.std()
    return media <= umbral


def normales(xyz, q, chunk):
    """Normal = autovector de menor autovalor de la covarianza de q vecinos."""
    arbol = cKDTree(xyz)
    n = np.empty((len(xyz), 3), np.float32)
    for i in range(0, len(xyz), chunk):
        _, idx = arbol.query(xyz[i:i + chunk], k=q, workers=-1)
        vec = xyz[idx]                                  # (b, q, 3)
        vec = vec - vec.mean(axis=1, keepdims=True)
        cov = np.einsum("bki,bkj->bij", vec, vec) / q
        _, autovec = np.linalg.eigh(cov)                # autovalores ascendentes
        n[i:i + chunk] = autovec[:, :, 0]
    # Orientacion consistente: hacia arriba (las verticales quedan como salen).
    n[n[:, 2] < 0] *= -1
    return n


def main():
    args = parse_args()
    t0 = time.time()
    meta = {"input": [str(p) for p in args.input], "voxel": args.voxel,
            "q": args.q, "lambda": args.lam, "prevoxel": args.prevoxel}

    xyz, rgb = leer_las(args.input, args.chunk)
    offset = (xyz.min(axis=0) + xyz.max(axis=0)) / 2
    xyz -= offset
    meta["offset_utm"] = offset.tolist()
    meta["n_crudo"] = int(len(xyz))
    print(f"[lectura] {len(xyz):,} puntos  ({time.time() - t0:.0f}s)", flush=True)

    if args.prevoxel > 0:
        xyz, rgb = voxelizar(xyz, rgb, args.prevoxel)
        meta["n_prevoxel"] = int(len(xyz))
        print(f"[prevoxel] {len(xyz):,} puntos  ({time.time() - t0:.0f}s)", flush=True)

    ok = sor(xyz, args.q, args.lam, args.chunk)
    xyz, rgb = xyz[ok], rgb[ok]
    meta["n_sor"] = int(len(xyz))
    print(f"[sor] quedan {len(xyz):,} ({(~ok).sum():,} descartados)  "
          f"({time.time() - t0:.0f}s)", flush=True)

    xyz, rgb = voxelizar(xyz, rgb, args.voxel)
    meta["n_voxel"] = int(len(xyz))
    print(f"[voxel] {len(xyz):,} puntos  ({time.time() - t0:.0f}s)", flush=True)

    nrm = normales(xyz, args.q, args.chunk // 5)
    print(f"[normales] listo  ({time.time() - t0:.0f}s)", flush=True)

    salida = Path(args.output)
    salida.parent.mkdir(parents=True, exist_ok=True)
    datos = np.hstack([xyz.astype(np.float32), rgb, nrm]).astype(np.float32)
    np.save(salida.with_suffix(".npy"), datos)
    meta["segundos"] = round(time.time() - t0, 1)
    salida.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"[ok] {salida.with_suffix('.npy')}  {datos.shape}", flush=True)


if __name__ == "__main__":
    main()
