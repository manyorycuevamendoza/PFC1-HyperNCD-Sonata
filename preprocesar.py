"""Modulo 1 del pipeline: preprocesamiento de un levantamiento LAS.

Lee una o varias nubes .las, las une en una sola seccion S_j y entrega la
nube limpia y voxelizada que consume extraer_features.py (Modulo 3).

Pasos (capitulo 2, Modulo 1):
    1. Lectura por bloques con laspy: XYZ + RGB. Las coordenadas se centran
       restando el centro de la seccion (vienen en UTM, ~10^6 m).
    2. Limpieza de ruido por SOR con q vecinos y umbral lambda (Open3D).
    3. Voxelizacion: promedio de XYZ y RGB por voxel (Open3D).
    4. Normales por PCA de los mismos q vecinos (Open3D).

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
import open3d as o3d


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
                    help="puntos por bloque al leer el LAS")
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


def a_open3d(xyz, rgb):
    nube = o3d.geometry.PointCloud()
    nube.points = o3d.utility.Vector3dVector(xyz)
    nube.colors = o3d.utility.Vector3dVector(rgb / 255.0)  # Open3D usa 0-1
    return nube


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
    nube = a_open3d(xyz, rgb)
    del xyz, rgb

    if args.prevoxel > 0:
        nube = nube.voxel_down_sample(args.prevoxel)
        meta["n_prevoxel"] = len(nube.points)
        print(f"[prevoxel] {len(nube.points):,} puntos  ({time.time() - t0:.0f}s)", flush=True)

    # SOR: descarta puntos cuya distancia media a sus q vecinos supera
    # mu + lambda * sigma.
    n_antes = len(nube.points)
    nube, _ = nube.remove_statistical_outlier(nb_neighbors=args.q,
                                              std_ratio=args.lam)
    meta["n_sor"] = len(nube.points)
    print(f"[sor] quedan {len(nube.points):,} ({n_antes - len(nube.points):,} "
          f"descartados)  ({time.time() - t0:.0f}s)", flush=True)

    # Voxelizacion: un representante por voxel (promedio de posicion y color).
    nube = nube.voxel_down_sample(args.voxel)
    meta["n_voxel"] = len(nube.points)
    print(f"[voxel] {len(nube.points):,} puntos  ({time.time() - t0:.0f}s)", flush=True)

    # Normales: autovector de menor autovalor de la covarianza de q vecinos,
    # orientadas hacia arriba para que sean consistentes.
    nube.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=args.q))
    nube.orient_normals_to_align_with_direction([0.0, 0.0, 1.0])
    print(f"[normales] listo  ({time.time() - t0:.0f}s)", flush=True)

    salida = Path(args.output)
    salida.parent.mkdir(parents=True, exist_ok=True)
    datos = np.hstack([np.asarray(nube.points),
                       np.asarray(nube.colors) * 255.0,
                       np.asarray(nube.normals)]).astype(np.float32)
    np.save(salida.with_suffix(".npy"), datos)
    meta["segundos"] = round(time.time() - t0, 1)
    salida.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"[ok] {salida.with_suffix('.npy')}  {datos.shape}", flush=True)


if __name__ == "__main__":
    main()
