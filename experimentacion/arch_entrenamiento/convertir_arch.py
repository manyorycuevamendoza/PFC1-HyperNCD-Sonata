"""Convierte ArCH_labelled (.txt) a bloques .npz listos para entrenar HyperNCD.

Por escena: lee el .txt, voxeliza a 5 cm (promedio de XYZ y RGB, etiqueta por mayoría de votos),
y corta en bloques de TILE x TILE metros (en X e Y, toda la altura). Un bloque = una muestra de entrenamiento.
Formato ArCH: x y z r g b label Nx Ny Nz. Etiquetas: 0 arch,1 column,2 moldings,3 floor,4 door_window,
5 wall,6 stairs,7 vault,8 roof,9 other (other se guarda como 9; el config la manda a "ignorar").

Salida:  <salida>/<Training|Test>/<escena>__<i>_<j>.npz   con  xyz (float32, relativo al bloque),
         rgb (uint8), label (int8)     y   <salida>/resumen_bloques.json
Es reanudable: salta las escenas que ya tienen su marca <escena>.listo.
Uso: python convertir_arch.py --entrada .../ArCH_labelled --salida .../tiles [--voxel 0.05] [--tile 10] [--minpts 3000]
"""
import argparse, glob, json, os, time
import numpy as np, pandas as pd


def leer(ruta, bloque=3_000_000):
    xs = []
    with open(ruta) as f:
        p = f.readline().split()
    cab = not all(t.replace(".", "", 1).replace("-", "", 1).replace("e", "", 1).isdigit() for t in p[:3])
    for ch in pd.read_csv(ruta, sep=r"\s+", header=None, skiprows=1 if cab else 0, usecols=[0, 1, 2, 3, 4, 5, 6],
                          dtype=np.float32, chunksize=bloque, on_bad_lines="skip"):
        xs.append(ch.dropna().to_numpy(np.float32))
    return np.concatenate(xs)


def voxelizar(a, vox):
    """a: (N,7) x y z r g b label -> xyz, rgb, label por voxel (etiqueta por mayoría)."""
    xyz = a[:, :3].astype(np.float64)
    mn = xyz.min(0)
    ijk = np.floor((xyz - mn) / vox).astype(np.int64)
    dim = ijk.max(0) + 1
    clave = (ijk[:, 0] * dim[1] + ijk[:, 1]) * dim[2] + ijk[:, 2]
    uniq, inv, cnt = np.unique(clave, return_inverse=True, return_counts=True)
    n = len(uniq)
    out = np.zeros((n, 6), np.float64)
    for k in range(3):
        out[:, k] = np.bincount(inv, weights=xyz[:, k], minlength=n) / cnt
    for k in range(3):
        out[:, 3 + k] = np.bincount(inv, weights=a[:, 3 + k].astype(np.float64), minlength=n) / cnt
    lab = np.clip(a[:, 6].astype(np.int64), 0, 9)
    votos = np.zeros((n, 10), np.int32)
    np.add.at(votos, (inv, lab), 1)
    return out[:, :3].astype(np.float32), np.clip(out[:, 3:], 0, 255).astype(np.uint8), votos.argmax(1).astype(np.int8)


def cortar(xyz, rgb, lab, tile, minpts, salida, nombre):
    mn = xyz[:, :2].min(0)
    ij = np.floor((xyz[:, :2] - mn) / tile).astype(np.int64)
    claves = ij[:, 0] * 100000 + ij[:, 1]
    orden = np.argsort(claves, kind="stable")
    cs = claves[orden]
    cortes = np.flatnonzero(np.diff(cs)) + 1
    info = []
    for grupo in np.split(orden, cortes):
        if len(grupo) < minpts:
            continue
        i, j = ij[grupo[0]]
        p = xyz[grupo]
        np.savez_compressed(f"{salida}/{nombre}__{i}_{j}.npz", xyz=(p - p.min(0)).astype(np.float32),
                            rgb=rgb[grupo], label=lab[grupo])
        info.append({"bloque": f"{nombre}__{i}_{j}", "puntos": int(len(grupo)),
                     "por_clase": np.bincount(lab[grupo].astype(np.int64), minlength=10).tolist()})
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entrada", required=True); ap.add_argument("--salida", required=True)
    ap.add_argument("--voxel", type=float, default=0.05); ap.add_argument("--tile", type=float, default=10.0)
    ap.add_argument("--minpts", type=int, default=3000)
    a = ap.parse_args()
    resumen = []
    for sub in ("Training", "Test"):
        os.makedirs(f"{a.salida}/{sub}", exist_ok=True)
        for ruta in sorted(glob.glob(f"{a.entrada}/{sub}/*.txt")):
            nombre = os.path.splitext(os.path.basename(ruta))[0]
            marca = f"{a.salida}/{sub}/{nombre}.listo.json"
            if os.path.exists(marca):
                resumen += json.load(open(marca)); print("ya listo:", sub, nombre, flush=True); continue
            t = time.time()
            datos = leer(ruta); n0 = len(datos)
            xyz, rgb, lab = voxelizar(datos, a.voxel); del datos
            info = cortar(xyz, rgb, lab, a.tile, a.minpts, f"{a.salida}/{sub}", nombre)
            for r in info:
                r["escena"], r["conjunto"] = nombre, sub
            json.dump(info, open(marca, "w"))
            resumen += info
            print(f"{sub}/{nombre}: {n0:,} -> {len(xyz):,} voxeles, {len(info)} bloques ({time.time() - t:.0f} s)", flush=True)
    json.dump(resumen, open(f"{a.salida}/resumen_bloques.json", "w"))
    for sub in ("Training", "Test"):
        r = [x for x in resumen if x["conjunto"] == sub]
        print(sub, len(r), "bloques,", f"{sum(x['puntos'] for x in r):,}", "puntos")


if __name__ == "__main__":
    main()
