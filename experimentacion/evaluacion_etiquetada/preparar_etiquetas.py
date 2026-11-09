"""Prepara una escena etiquetada por archivos: un .las por pieza y el NOMBRE del archivo da la clase.

Ejemplo (carpeta `listo`): arco1.las ... arco89.las -> clase "arco"; columna1.las -> "columna";
cupula.las, pared1.las ..., techo.las, otro.las. La clase es el nombre sin digitos ni extension.

Dos pasos:

1. unir     junta todos los .las en UN solo .las (para pasarlo por el Modulo 1) y guarda las
            etiquetas de cada punto crudo en un .npz.
2. mapear   el Modulo 1 filtra (SOR) y voxeliza, asi que su .npy tiene otros puntos. Aqui cada
            punto del .npy hereda la etiqueta del punto crudo mas cercano (si esta a mas de
            --max-dist metros queda sin etiqueta = -1). Guarda <npy>_etiquetas.npy y un .json
            con el control de calidad (distancias).

Uso:
    python experimentacion/evaluacion_etiquetada/preparar_etiquetas.py unir \
        --carpeta C:/ruta/listo --salida-las work/escena.las --salida-etiq work/escena_etiquetas.npz
    python preprocesar.py --input work/escena.las --output work/escena
    python experimentacion/evaluacion_etiquetada/preparar_etiquetas.py mapear \
        --npy work/escena.npy --json work/escena.json --etiquetas work/escena_etiquetas.npz
"""
import argparse
import json
import re
import unicodedata
from pathlib import Path

import laspy
import numpy as np


def nombre_clase(ruta):
    """arco89.las -> 'arco'; Cúpula2.las -> 'cupula'."""
    base = Path(ruta).stem
    base = unicodedata.normalize("NFKD", base).encode("ascii", "ignore").decode()
    base = re.sub(r"[\d_\-\s]+$", "", base.lower())
    return base or "sin_nombre"


def unir(a):
    archivos = sorted(Path(a.carpeta).rglob("*.las"))
    if not archivos:
        raise SystemExit(f"No hay .las en {a.carpeta}")
    nombres = sorted({nombre_clase(f) for f in archivos})
    # "otro" va al final (clase de fondo); el resto en orden alfabetico
    clases = [c for c in nombres if c != "otro"] + (["otro"] if "otro" in nombres else [])
    print(f"{len(archivos)} archivos, {len(clases)} clases: {clases}")

    # "otro" primero: si un punto esta en "otro" y en otra pieza (en el dataset `listo`, otro.las trae
    # TODA la escena), la clase especifica gana sobre "otro".
    archivos = sorted(archivos, key=lambda f: (nombre_clase(f) != "otro", f.name))
    xyz, rgb, lab, archivo_id = [], [], [], []
    for i, f in enumerate(archivos):
        las = laspy.read(f)
        p = np.c_[las.x, las.y, las.z]
        ok = np.isfinite(p).all(1)
        xyz.append(p[ok])
        rgb.append(np.c_[las.red, las.green, las.blue][ok])
        lab.append(np.full(ok.sum(), clases.index(nombre_clase(f)), np.uint8))
        archivo_id.append(np.full(ok.sum(), i, np.uint16))
        print(f"  {f.name:22s} {ok.sum():>10,d} puntos -> {nombre_clase(f)}")
    xyz, rgb = np.concatenate(xyz), np.concatenate(rgb)
    lab, archivo_id = np.concatenate(lab), np.concatenate(archivo_id)

    # puntos repetidos entre piezas: se conserva la ULTIMA copia (las clases especificas van despues de "otro")
    clave = np.round(xyz / 0.001).astype(np.int64)
    _, rev = np.unique(clave[::-1], axis=0, return_index=True)
    mantener = np.sort(len(clave) - 1 - rev)
    if len(mantener) < len(xyz):
        print(f"\n{len(xyz) - len(mantener):,} puntos aparecen en mas de una pieza; gana la clase especifica sobre 'otro'")
        i_otro = clases.index("otro") if "otro" in clases else -1
        esp = lab != i_otro
        pares = np.unique(np.c_[clave[esp], lab[esp]], axis=0)
        claves = np.unique(clave[esp], axis=0)
        if len(pares) > len(claves):
            print(f"AVISO: {len(pares) - len(claves):,} puntos estan en DOS clases especificas distintas "
                  "(se conserva una al azar); revisa el solapamiento entre piezas")
        xyz, rgb, lab, archivo_id = xyz[mantener], rgb[mantener], lab[mantener], archivo_id[mantener]

    ref = laspy.read(archivos[0]).header
    cab = laspy.LasHeader(point_format=ref.point_format.id, version=f"{ref.version.major}.{ref.version.minor}")
    cab.scales = ref.scales
    cab.offsets = xyz.min(0)
    las = laspy.LasData(cab)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    las.red, las.green, las.blue = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    Path(a.salida_las).parent.mkdir(parents=True, exist_ok=True)
    las.write(a.salida_las)
    np.savez_compressed(a.salida_etiq, xyz=xyz, label=lab, clases=np.array(clases),
                        archivos=np.array([f.name for f in archivos]), archivo_id=archivo_id)
    cuenta = np.bincount(lab, minlength=len(clases))
    print(f"\nTotal {len(xyz):,} puntos -> {a.salida_las}")
    for c, n in zip(clases, cuenta):
        print(f"  {c:10s} {n:>10,d}  ({100 * n / len(lab):.1f} %)")


def mapear(a):
    from scipy.spatial import cKDTree
    z = np.load(a.etiquetas, allow_pickle=False)
    meta = json.loads(Path(a.json).read_text())
    S = np.load(a.npy, mmap_mode="r")
    pts = np.asarray(S[:, :3], np.float64) + np.array(meta["offset_utm"])     # de vuelta a coordenadas del LAS
    d, i = cKDTree(z["xyz"]).query(pts, workers=-1)
    lab = z["label"][i].astype(np.int8)
    lab[d > a.max_dist] = -1
    salida = Path(a.npy).with_name(Path(a.npy).stem + "_etiquetas.npy")
    np.save(salida, lab)
    clases = [str(c) for c in z["clases"]]
    cuenta = {c: int((lab == k).sum()) for k, c in enumerate(clases)}
    informe = {
        "n_puntos": int(len(lab)), "clases": clases, "puntos_por_clase": cuenta,
        "sin_etiqueta": int((lab < 0).sum()), "max_dist_m": a.max_dist,
        "dist_mediana_m": float(np.median(d)), "dist_p95_m": float(np.percentile(d, 95)),
        "dist_max_m": float(d.max()),
    }
    Path(str(salida).replace(".npy", ".json")).write_text(json.dumps(informe, indent=2, ensure_ascii=False))
    print(f"{len(lab):,} puntos etiquetados -> {salida}")
    print(f"distancia al punto crudo: mediana {informe['dist_mediana_m']*100:.1f} cm, "
          f"p95 {informe['dist_p95_m']*100:.1f} cm; sin etiqueta (> {a.max_dist} m): {informe['sin_etiqueta']:,}")
    for c, n in cuenta.items():
        print(f"  {c:10s} {n:>10,d}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    u = sp.add_parser("unir")
    u.add_argument("--carpeta", required=True)
    u.add_argument("--salida-las", dest="salida_las", required=True)
    u.add_argument("--salida-etiq", dest="salida_etiq", required=True)
    u.set_defaults(f=unir)
    m = sp.add_parser("mapear")
    m.add_argument("--npy", required=True)
    m.add_argument("--json", required=True, help=".json de preprocesar.py (trae offset_utm)")
    m.add_argument("--etiquetas", required=True, help=".npz de `unir`")
    m.add_argument("--max-dist", dest="max_dist", type=float, default=0.10)
    m.set_defaults(f=mapear)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
