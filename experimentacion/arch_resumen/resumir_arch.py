"""Resumen de ArCH_labelled: puntos, extensión y puntos por clase de cada escena.
Lee los .txt por bloques (no carga nada entero en memoria) y guarda un json por escena
(reanudable) y un resumen_arch.json + resumen_arch.md.
Formato ArCH: x y z r g b label Nx Ny Nz (separado por espacios)."""
import json, os, sys, glob, time
import numpy as np, pandas as pd

CLASES = ["arch", "column", "moldings", "floor", "door_window", "wall", "stairs", "vault", "roof", "other"]

def resumir(ruta, bloque=2_000_000):
    n, mn, mx = 0, np.full(3, np.inf), np.full(3, -np.inf)
    cuenta, fuera, filas_malas = np.zeros(10, np.int64), 0, 0
    with open(ruta) as f:
        primera = f.readline().split()
    cab = not all(t.replace(".", "", 1).replace("-", "", 1).replace("e", "", 1).isdigit() for t in primera[:3])
    ncol = len(primera)
    for ch in pd.read_csv(ruta, sep=r"\s+", header=None, skiprows=1 if cab else 0, usecols=[0, 1, 2, 6],
                          dtype=np.float64, chunksize=bloque, on_bad_lines="skip"):
        ch = ch.dropna()
        xyz, lab = ch.iloc[:, :3].to_numpy(), ch.iloc[:, 3].to_numpy().astype(np.int64)
        n += len(ch); mn = np.minimum(mn, xyz.min(0)); mx = np.maximum(mx, xyz.max(0))
        ok = (lab >= 0) & (lab < 10); fuera += int((~ok).sum())
        cuenta += np.bincount(lab[ok], minlength=10)
    return {"archivo": os.path.basename(ruta), "bytes": os.path.getsize(ruta), "columnas": ncol, "cabecera": bool(cab),
            "puntos": int(n), "min": mn.round(3).tolist(), "max": mx.round(3).tolist(),
            "extension_m": (mx - mn).round(2).tolist(), "por_clase": {CLASES[i]: int(cuenta[i]) for i in range(10)},
            "etiquetas_fuera_de_0_9": fuera}

def main(carpeta, salida):
    os.makedirs(salida, exist_ok=True)
    res = []
    for sub in ("Training", "Test"):
        for ruta in sorted(glob.glob(f"{carpeta}/{sub}/*.txt")):
            cache = f"{salida}/{sub}__{os.path.basename(ruta)}.json"
            if os.path.exists(cache):
                r = json.load(open(cache))
            else:
                t = time.time(); print("leyendo", sub, os.path.basename(ruta), flush=True)
                r = resumir(ruta); r["conjunto"] = sub; r["segundos"] = round(time.time() - t)
                json.dump(r, open(cache, "w"), indent=1)
            res.append(r); print(f"  {r['puntos']:>11,} pts  ext {r['extension_m']}  ({r['segundos']} s)", flush=True)
    json.dump(res, open(f"{salida}/resumen_arch.json", "w"), indent=1)
    tot = {c: sum(r["por_clase"][c] for r in res) for c in CLASES}
    T = sum(tot.values())
    L = ["| Escena | Conj. | Puntos | Extensión (m) | " + " | ".join(CLASES) + " |", "|---|---|---|---|" + "---|" * 10]
    for r in res:
        p = r["puntos"] or 1
        L.append(f"| {r['archivo']} | {r['conjunto']} | {r['puntos']:,} | {r['extension_m']} | " +
                 " | ".join(f"{100 * r['por_clase'][c] / p:.1f}%" for c in CLASES) + " |")
    L.append(f"| **TOTAL** | | {T:,} | | " + " | ".join(f"{100 * tot[c] / T:.1f}%" for c in CLASES) + " |")
    L.append("\nEscenas donde aparece cada clase: " + ", ".join(f"{c}={sum(r['por_clase'][c] > 0 for r in res)}" for c in CLASES))
    open(f"{salida}/resumen_arch.md", "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
