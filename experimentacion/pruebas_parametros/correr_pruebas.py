"""Pruebas de parametros del Modulo 1 (P0 a P3) sobre la escena etiquetada, en CPU.

Cada prueba cambia UN parametro de `preprocesar.py` y mide la salida contra la corrida de referencia
(q = 20, lambda = 2, voxel = 5 cm, normales con 20 vecinos, bloque de 5e6). Las pruebas P4 y P5 y el ARI y NMI de
P2 necesitan la GPU (Sonata) y no se corren aqui.

Medidas (en una muestra fija de 200 000 puntos de la referencia, emparejada por vecino mas cercano):
  r            rugosidad, distancia media de cada punto al plano de sus 20 vecinos, en milimetros
  disp         dispersion angular, angulo entre la normal de cada punto y la normal media de sus 20 vecinos, en grados
  ralas        el decil mas disperso de la referencia (mayor distancia media a sus 20 vecinos)
  planas       la mitad de la referencia con menor curvatura, lambda3/(lambda1+lambda2+lambda3)
  aristas      el decil con mayor curvatura
  ret_g        porcentaje de puntos que conserva la rejilla de Sonata de lado g (celdas ocupadas sobre puntos)

Uso (desde la raiz del repo):
    python experimentacion/pruebas_parametros/correr_pruebas.py --las <escena.las> --trabajo <carpeta temporal> \
        --salida experimentacion/pruebas_parametros/resultados
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("--las", required=True)
ap.add_argument("--trabajo", required=True)
ap.add_argument("--salida", required=True)
ap.add_argument("--muestra", type=int, default=200000)
ap.add_argument("--pruebas", nargs="*", default=["P0", "P1", "P2", "P3"])
a = ap.parse_args()

RAIZ = Path(__file__).resolve().parents[2]
TRABAJO = Path(a.trabajo)
TRABAJO.mkdir(parents=True, exist_ok=True)
SALIDA = Path(a.salida)
SALIDA.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)

COLUMNAS = ["prueba", "condicion", "q", "q_normales", "lam", "voxel", "chunk", "n_sor", "n_voxel", "pct_descartado",
            "segundos", "r_mm_global", "r_mm_ralas", "disp_grados_global", "disp_grados_ralas",
            "disp_grados_planas", "disp_grados_aristas", "ret_g002_pct", "ret_g004_pct", "puntos_emparejados",
            "S_identica_a_ref"]


def correr(nombre, q=20, qn=None, lam=2.0, voxel=0.05, chunk=5_000_000):
    out = TRABAJO / nombre
    orden = [sys.executable, str(RAIZ / "preprocesar.py"), "--input", a.las, "--output", str(out),
             "--q", str(q), "--lam", str(lam), "--voxel", str(voxel), "--chunk", str(chunk)]
    if qn is not None:
        orden += ["--q-normales", str(qn)]
    t = time.time()
    r = subprocess.run(orden, capture_output=True, text=True)
    (SALIDA / "logs").mkdir(exist_ok=True)
    (SALIDA / "logs" / f"{nombre}.log").write_text(r.stdout + r.stderr, encoding="utf-8")
    if r.returncode != 0:
        raise RuntimeError(f"{nombre} fallo, ver logs/{nombre}.log")
    meta = json.loads(out.with_suffix(".json").read_text())
    meta["seg_total"] = round(time.time() - t, 1)
    return np.load(out.with_suffix(".npy")), meta


def vecinos(xyz, idx, k=20):
    tree = cKDTree(xyz)
    d, nb = tree.query(xyz[idx], k=k + 1)
    return d[:, 1:], nb[:, 1:]


def geometria(xyz, idx, nb):
    """Rugosidad y curvatura de cada punto idx a partir de sus vecinos nb."""
    P = xyz[nb].astype(np.float64)
    mu = P.mean(axis=1, keepdims=True)
    C = P - mu
    cov = np.einsum("nki,nkj->nij", C, C) / P.shape[1]
    w, v = np.linalg.eigh(cov)
    n3 = v[:, :, 0]                         # autovector del menor autovalor
    r = np.abs(np.einsum("ni,ni->n", xyz[idx].astype(np.float64) - mu[:, 0, :], n3)) * 1000.0
    curv = w[:, 0] / np.clip(w.sum(axis=1), 1e-12, None)
    return r, curv


def dispersion(normales, idx, nb):
    m = normales[nb].mean(axis=1)
    m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-12
    n = normales[idx]
    cos = np.abs(np.einsum("ni,ni->n", n, m)).clip(0, 1)
    return np.degrees(np.arccos(cos))


def retencion(xyz, g):
    celdas = np.floor((xyz - xyz.min(axis=0)) / g).astype(np.int64)
    u = np.unique(celdas, axis=0)
    return 100.0 * len(u) / len(xyz)


# ---------- referencia ----------
print("referencia", flush=True)
S0, m0 = correr("ref")
xyz0, nor0 = S0[:, :3], S0[:, 6:9]
muestra = rng.choice(len(S0), size=min(a.muestra, len(S0)), replace=False)
d0, nb0 = vecinos(xyz0, muestra)
r0, curv0 = geometria(xyz0, muestra, nb0)
dens0 = d0.mean(axis=1)
ralas = dens0 >= np.percentile(dens0, 90)
planas = curv0 <= np.percentile(curv0, 50)
aristas = curv0 >= np.percentile(curv0, 90)
ref_pts = xyz0[muestra]
n_crudo = m0["n_crudo"]


def medir(S, meta, prueba, condicion, q, qn, lam, voxel, chunk, ref=False):
    xyz, nor = S[:, :3], S[:, 6:9]
    tree = cKDTree(xyz)
    d, j = tree.query(ref_pts, k=1)
    ok = d <= 0.05
    # un punto de la condicion puede emparejar con varios de la referencia: se deja uno por pareja
    jj = j[ok]
    uj, primero = np.unique(jj, return_index=True)
    sel = np.where(ok)[0][primero]
    _, nb = cKDTree(xyz).query(xyz[uj], k=21)
    nb = nb[:, 1:]
    r, _ = geometria(xyz, uj, nb)
    disp = dispersion(nor, uj, nb)
    f = lambda v, m: float(np.mean(v[m[sel]])) if m[sel].any() else float("nan")
    # los indicadores se calculan en los puntos de la condicion; la marca (ralas, planas, aristas) viene de la referencia
    fila = {
        "prueba": prueba, "condicion": condicion, "q": q, "q_normales": qn if qn is not None else q, "lam": lam,
        "voxel": voxel, "chunk": chunk, "n_sor": meta["n_sor"], "n_voxel": meta["n_voxel"],
        "pct_descartado": round(100.0 * (1 - meta["n_sor"] / n_crudo), 2), "segundos": meta["seg_total"],
        "r_mm_global": round(float(r.mean()), 3), "r_mm_ralas": round(f(r, ralas), 3),
        "disp_grados_global": round(float(disp.mean()), 3), "disp_grados_ralas": round(f(disp, ralas), 3),
        "disp_grados_planas": round(f(disp, planas), 3), "disp_grados_aristas": round(f(disp, aristas), 3),
        "ret_g002_pct": round(retencion(xyz, 0.02), 2), "ret_g004_pct": round(retencion(xyz, 0.04), 2),
        "puntos_emparejados": int(len(uj)),
        "S_identica_a_ref": "" if ref else (bool(S.shape == S0.shape and np.array_equal(S, S0))),
    }
    return fila


def escribir(nombre, filas):
    with open(SALIDA / f"{nombre}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNAS)
        w.writeheader()
        for f in filas:
            w.writerow({c: f.get(c, "") for c in COLUMNAS})
    print(f"{nombre}.csv  {len(filas)} filas", flush=True)


ref_fila = medir(S0, m0, "ref", "referencia", 20, None, 2.0, 0.05, 5_000_000, ref=True)

if "P0" in a.pruebas:
    filas = [dict(ref_fila, prueba="P0", condicion="chunk 5e6 (referencia)")]
    for c in (1_000_000, 20_000_000):
        S, m = correr(f"P0_chunk{c}", chunk=c)
        filas.append(medir(S, m, "P0", f"chunk {c:.0e}", 20, None, 2.0, 0.05, c))
        del S
    escribir("P0_bloque", filas)

if "P1" in a.pruebas:
    cond = [("sin filtrado", dict(q=20, lam=1e9)), ("q=10", dict(q=10)), ("q=20 (referencia)", dict(q=20)),
            ("q=30", dict(q=30)), ("q=40", dict(q=40)), ("lambda=1,5", dict(lam=1.5)), ("lambda=2,5", dict(lam=2.5)),
            ("lambda=3", dict(lam=3.0))]
    filas = []
    for nombre, kw in cond:
        if nombre.endswith("(referencia)"):
            filas.append(dict(ref_fila, prueba="P1", condicion=nombre)); continue
        S, m = correr("P1_" + nombre.replace("=", "").replace(" ", "_").replace(",", "p"), **kw)
        filas.append(medir(S, m, "P1", nombre, kw.get("q", 20), None, kw.get("lam", 2.0), 0.05, 5_000_000))
        del S
    escribir("P1_filtrado", filas)

if "P2" in a.pruebas:
    filas = []
    for v in (0.02, 0.03, 0.05, 0.10):
        if v == 0.05:
            filas.append(dict(ref_fila, prueba="P2", condicion="voxel 5 cm (referencia)")); continue
        S, m = correr(f"P2_voxel{int(v * 100)}", voxel=v)
        filas.append(medir(S, m, "P2", f"voxel {int(v * 100)} cm", 20, None, 2.0, v, 5_000_000))
        del S
    escribir("P2_voxel", filas)

if "P3" in a.pruebas:
    filas = []
    for qn in (10, 20, 30, 40):
        if qn == 20:
            filas.append(dict(ref_fila, prueba="P3", condicion="normales con 20 (referencia)")); continue
        S, m = correr(f"P3_qn{qn}", qn=qn)
        filas.append(medir(S, m, "P3", f"normales con {qn}", 20, qn, 2.0, 0.05, 5_000_000))
        del S
    escribir("P3_normales", filas)
print("listo", flush=True)
