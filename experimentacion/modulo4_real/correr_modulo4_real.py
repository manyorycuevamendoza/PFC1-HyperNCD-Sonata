"""Modulo 4 REAL sobre PL13 con f_sem (Modulo 2) y f_ssl (Modulo 3).

Une los descriptores de los dos modulos por punto original (los dos .npz traen
`inverse`, que lleva cada uno de los 803 099 puntos del .npy a su voxel), agrupa
los puntos en regiones y aplica las mismas formulas del Modulo 4 que usa la
simulacion (`modulo4_simulado/simular_modulo4.py`): afinidad S_b, hiperaristas,
prototipos y Sinkhorn semi-relajado.

Variantes comparadas (todas con K clases nuevas):
    M2       solo f_sem
    M3       solo f_ssl
    M4_base  F = [f_sem ; f_geo]   (linea base de HyperNCD)
    M4       F = [f_sem ; f_ssl]   (propuesta)

La Catedral NO tiene etiquetas, asi que no hay mIoU. Se mide:
    - tamanos de cada grupo y su perfil (altura, normal vertical, planaridad);
    - acuerdo entre variantes (ARI y NMI, ponderado por puntos);
    - tabla clase de M2 (13 cabezas) x grupo de cada variante;
    - silueta coseno de las regiones en el espacio de f_sem y de f_ssl.

Diferencia con la simulacion: las regiones son celdas de rejilla (--celda, 0,75 m)
y no DBSCAN, porque en una nave real todo esta conectado y DBSCAN daria una sola
region gigante.

Uso:
    python experimentacion/modulo4_real/correr_modulo4_real.py \
        --npy experimentacion/datos/modulo3_PL13/PL13_SCN0001.npy \
        --fsem cache/PL13_fsem.npz \
        --fssl experimentacion/datos/modulo3_PL13/PL13_feats.npz \
        --salida experimentacion/resultados/modulo4_real
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modulo4_simulado"))
from simular_modulo4 import normalizar, sinkhorn  # noqa: E402


def regiones_rejilla(xyz, celda):
    """Indice de region (0..R-1) por punto, a partir de celdas cubicas."""
    ij = np.floor((xyz - xyz.min(0)) / celda).astype(np.int64)
    clave = (ij[:, 0] * 100003 + ij[:, 1]) * 100019 + ij[:, 2]
    _, reg = np.unique(clave, return_inverse=True)
    return reg


def medias_por_region(reg, R, feats, idx, trozo=100_000):
    """Promedio por region de feats[idx] sin materializar la matriz completa."""
    n = len(reg)
    suma = np.zeros((R, feats.shape[1]), np.float64)
    for a in range(0, n, trozo):
        b = min(a + trozo, n)
        A = sp.csr_matrix((np.ones(b - a), (reg[a:b], np.arange(b - a))), shape=(R, b - a))
        suma += A @ feats[idx[a:b]].astype(np.float32)
    return (suma / np.bincount(reg, minlength=R)[:, None]).astype(np.float32)


def rasgos_region(xyz, normales, reg, R):
    """Linealidad, planaridad, dispersion y centroide por region (vectorizado)."""
    n = np.bincount(reg, minlength=R).astype(np.float64)
    c = np.stack([np.bincount(reg, xyz[:, k], R) for k in range(3)], 1) / n[:, None]
    d = xyz - c[reg]
    cov = np.stack([np.bincount(reg, d[:, i] * d[:, j], R) for i in range(3) for j in range(3)],
                   1).reshape(R, 3, 3) / n[:, None, None]
    ev = np.sort(np.linalg.eigvalsh(cov + 1e-12 * np.eye(3)), axis=1)[:, ::-1]
    l1, l2, l3 = ev[:, 0], ev[:, 1], ev[:, 2]
    g = np.c_[(l1 - l2) / l1, (l2 - l3) / l1, l3 / l1, c]
    nz = np.bincount(reg, np.abs(normales[:, 2]), R) / n
    return g.astype(np.float32), nz.astype(np.float32), c.astype(np.float32)


def m4_regiones(Freg, g, tam, K, rng, alpha=0.5, M=8, tau=1.0, gamma=1.0, iters=15):
    """Modulo 4 sobre descriptores de region ya promediados. Devuelve el grupo por region."""
    gz = (g - g.mean(0)) / (g.std(0) + 1e-9)
    Fn = normalizar(Freg)
    Ss = Fn @ Fn.T
    d2 = ((gz[:, None] - gz[None]) ** 2).sum(-1)
    Sg = np.exp(-d2 / (tau * gz.shape[1]))
    Sb = alpha * Sg + (1 - alpha) * Ss
    m = min(M, len(Freg) - 1)
    vec = np.argsort(-Sb, axis=1)[:, : m + 1]
    w = np.take_along_axis(np.clip(Sb, 0, None), vec, 1)
    Fh = normalizar((w[..., None] * Freg[vec]).sum(1) / (w.sum(1, keepdims=True) + 1e-12))
    P = [Fh[rng.integers(len(Fh))]]
    for _ in range(K - 1):
        dist = np.clip(1 - (Fh @ np.stack(P).T).max(1), 1e-12, None)
        P.append(Fh[rng.choice(len(Fh), p=dist / dist.sum())])
    P = np.stack(P)
    for _ in range(iters):
        s = sinkhorn(Fh @ normalizar(P).T, gamma=gamma, peso=tam)
        sw = s * tam[:, None]
        P = (sw.T @ Fh) / (sw.sum(0)[:, None] + 1e-12)
    return s.argmax(1)


def ari_ponderado(a, b, tam):
    """ARI/NMI donde cada region cuenta tantas veces como puntos tiene (en bloques)."""
    r = np.repeat(np.arange(len(a)), np.maximum(1, (tam / tam.min()).astype(int)))
    return adjusted_rand_score(a[r], b[r]), normalized_mutual_info_score(a[r], b[r])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npy", required=True)
    ap.add_argument("--fsem", required=True)
    ap.add_argument("--fssl", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--celda", type=float, default=0.75)
    ap.add_argument("--k", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--min-puntos", type=int, default=30)
    ap.add_argument("--semillas", type=int, default=3)
    ap.add_argument("--todos", action="store_true",
                    help="agrupa TODOS los puntos, no solo los que M2 marca como novedosos. Necesario para comparar "
                         "con el M2 de SemanticPOSS (que marcaba ~99 %% como novedoso) cuando el M2 nuevo (p. ej. ArCH) "
                         "solo marca ~13 %%")
    args = ap.parse_args()
    sal = Path(args.salida)
    sal.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    S = np.load(args.npy, mmap_mode="r")
    xyz = np.asarray(S[:, :3], np.float64)
    normales = np.asarray(S[:, 6:9], np.float32)
    z2 = np.load(args.fsem)
    z3 = np.load(args.fssl)
    inv2, inv3 = z2["inverse"], z3["inverse"]
    assert len(inv2) == len(inv3) == len(xyz), "los .npz no corresponden a este .npy"
    off = z3["coord"][inv3] - xyz
    # el voxel de M3 (2 cm) representa a sus puntos con uno solo: error <= tamano de voxel
    assert np.abs(off - np.median(off, 0)).max() < float(z3["grid_size"]) + 1e-3, \
        "M3 no corresponde a este .npy"
    assert np.abs(z2["coord_m1"][inv2] - xyz).max() < 0.06, "M2 no corresponde a este .npy"
    f_sem, f_ssl = z2["feat"], z3["feat"]
    pred_m2 = z2["pred_label"][inv2]
    nov_pt = z2["es_novedosa"][inv2] == 1
    print(f"puntos {len(xyz):,}  novedosos segun M2 {nov_pt.mean():.1%}", flush=True)
    if args.todos:
        nov_pt = np.ones(len(xyz), bool)
        print("--todos: se agrupan todos los puntos", flush=True)
    NH = int(z2["logits_lab"].shape[1] + z2["logits_unlab"].shape[1])   # cabezas de M2 (13 en SemanticPOSS, 9 en ArCH)
    NH = max(NH, int(pred_m2.max()) + 1)

    # regiones solo con puntos novedosos (como el Modulo 4: las conocidas ya tienen cabeza)
    idx = np.flatnonzero(nov_pt)
    reg0 = regiones_rejilla(xyz[idx], args.celda)
    tam0 = np.bincount(reg0)
    keep = tam0 >= args.min_puntos
    remap = np.cumsum(keep) - 1
    ok_pt = keep[reg0]
    idx, reg = idx[ok_pt], remap[reg0[ok_pt]]
    R = int(keep.sum())
    tam = np.bincount(reg, minlength=R).astype(np.float64)
    print(f"regiones {R:,} (celda {args.celda} m, >= {args.min_puntos} puntos); "
          f"cubren {len(idx)/nov_pt.sum():.1%} de los novedosos", flush=True)

    Fsem = medias_por_region(reg, R, f_sem, inv2[idx])
    Fssl = medias_por_region(reg, R, f_ssl, inv3[idx])
    g, nz, cen = rasgos_region(xyz[idx], normales[idx], reg, R)
    z_rel = cen[:, 2] - np.percentile(cen[:, 2], 1)
    print(f"descriptores listos ({time.time()-t0:.0f}s)", flush=True)

    # clase de M2 dominante por region y distribucion de clases de M2 por punto
    clases_m2 = np.unique(pred_m2[idx])
    cm2 = np.zeros((R, NH))
    np.add.at(cm2, (reg, pred_m2[idx]), 1)
    m2_reg = cm2.argmax(1)

    variantes = {
        "M2": Fsem,
        "M3": Fssl,
        "M4_base": np.hstack([normalizar(Fsem), normalizar(g[:, :3])]),
        "M4": np.hstack([normalizar(Fsem), normalizar(Fssl)]),
    }
    salida = {"regiones": R, "celda_m": args.celda, "puntos_novedosos": int(nov_pt.sum()),
              "cobertura": float(len(idx) / nov_pt.sum()), "K": {}}
    etiquetas = {}
    for K in args.k:
        res = {}
        for nombre, F in variantes.items():
            sems = [m4_regiones(F, g, tam, K, np.random.default_rng(s)) for s in range(args.semillas)]
            # estabilidad entre semillas y eleccion de la corrida mas central
            aris = [[ari_ponderado(a, b, tam)[0] for b in sems] for a in sems]
            centro = int(np.argmax(np.mean(aris, 1)))
            lab = sems[centro]
            estab = float(np.mean([aris[i][j] for i in range(len(sems)) for j in range(len(sems)) if i != j])) if len(sems) > 1 else 1.0
            res[nombre] = lab
            salida["K"].setdefault(K, {}).setdefault(nombre, {})["estabilidad_ari"] = estab
        etiquetas[K] = res

        rng = np.random.default_rng(0)
        sub = rng.choice(R, min(R, 4000), replace=False)
        for nombre, lab in res.items():
            d = salida["K"][K][nombre]
            d["puntos_por_grupo"] = [int(tam[lab == k].sum()) for k in range(K)]
            d["regiones_por_grupo"] = [int((lab == k).sum()) for k in range(K)]
            d["z_media_m"] = [float(np.average(z_rel[lab == k], weights=tam[lab == k])) if (lab == k).any() else None for k in range(K)]
            d["normal_vertical"] = [float(np.average(nz[lab == k], weights=tam[lab == k])) if (lab == k).any() else None for k in range(K)]
            d["planaridad"] = [float(np.average(g[lab == k, 1], weights=tam[lab == k])) if (lab == k).any() else None for k in range(K)]
            if len(np.unique(lab[sub])) > 1:
                d["silueta_fsem"] = float(silhouette_score(normalizar(Fsem[sub]), lab[sub], metric="cosine"))
                d["silueta_fssl"] = float(silhouette_score(normalizar(Fssl[sub]), lab[sub], metric="cosine"))
        nombres = list(res)
        acu = {}
        for i, a in enumerate(nombres):
            for b in nombres[i + 1:]:
                ari, nmi = ari_ponderado(res[a], res[b], tam)
                acu[f"{a}~{b}"] = {"ari": float(ari), "nmi": float(nmi)}
        salida["K"][K]["acuerdo"] = acu
        # clase de M2 x grupo de cada variante (puntos), solo clases con >= 0,5 % de puntos
        tabla = {}
        for nombre, lab in res.items():
            T = np.zeros((NH, K))
            np.add.at(T, (m2_reg, lab), tam)
            tabla[nombre] = T
        salida["K"][K]["tabla_m2"] = {n: T.astype(int).tolist() for n, T in tabla.items()}
        print(f"K={K} listo ({time.time()-t0:.0f}s)", flush=True)

    np.savez_compressed(sal / "grupos_PL13.npz", idx_puntos=idx.astype(np.int32),
                        region=reg.astype(np.int32), cen=cen,
                        **{f"K{K}_{n}": etiquetas[K][n].astype(np.int8)
                           for K in args.k for n in etiquetas[K]})
    (sal / "modulo4_real_PL13.json").write_text(json.dumps(salida, indent=2))
    print(f"[ok] {sal}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
