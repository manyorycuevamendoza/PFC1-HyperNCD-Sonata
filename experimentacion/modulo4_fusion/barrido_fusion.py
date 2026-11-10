"""Barrido de formas DETERMINISTAS de fusionar f_sem (M2) y f_ssl (M3) en el Modulo 4, sobre la escena etiquetada.

Pregunta: con un M2 que ya no colapsa, ¿hay una fusion mejor que pegar f_sem y f_ssl normalizados (la del M4 actual)?
Todo trabaja sobre descriptores POR REGION (celdas de 0,75 m, >= 30 puntos), asi que tras un calculo unico (cache) cada
configuracion tarda ~1 s. No se usan etiquetas para agrupar; las etiquetas solo sirven para evaluar.

Ejes del barrido
  pre    raw      descriptores tal cual (lo que hace el M4 actual)
         center   restar la media (ponderada por puntos) de cada fuente
         pca      centrar + PCA con blanqueo (f_sem a 32 dim, f_ssl a 64 dim): ninguna dimension domina
  modo   lineal   S = a*cos_sem + (1-a)*cos_ssl           (equivale a pegar los vectores con pesos; a=0,5 es el M4 actual)
         prod     media geometrica ponderada de (1+cos)/2 (exige que las dos fuentes esten de acuerdo)
         min      minimo de los dos cosenos
         conf     peso por region segun la confianza de M2 (mas f_sem donde sus cabezas estan seguras)
         snf      Similarity Network Fusion (Wang 2014), 2 vistas
  a      peso de f_sem (0,1 a 0,9)
  a_geo  peso de la afinidad geometrica en la mezcla S_b (el M4 actual usa 0,5)

Protocolo honesto (para no sobreajustar con UNA escena): la escena se parte en dos mitades espaciales A y B. El
agrupamiento usa todas las regiones (no usa etiquetas). Se ELIGE la configuracion con el mIoU de una mitad y se REPORTA
en la otra (y viceversa); el promedio es la estimacion honesta. Se reporta tambien la media y la desviacion sobre semillas.

Uso (desde la raiz del repo):
    python experimentacion/modulo4_fusion/barrido_fusion.py --nombre arch_original \
        --npy <capilla.npy> --fsem <fsem.npz> --fssl <feats.npz> --etiquetas <etiquetas.npy> --info <etiquetas.json> \
        --salida experimentacion/modulo4_fusion/resultados
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parents[0] / "modulo4_real"))
sys.path.insert(0, str(AQUI.parents[0] / "modulo4_simulado"))
from correr_modulo4_real import (m4_regiones, medias_por_region, rasgos_region,  # noqa: E402
                                 regiones_rejilla)
from simular_modulo4 import normalizar, sinkhorn  # noqa: E402


# ---------------------------------------------------------------- cache de regiones
def construir_cache(a, ruta):
    t0 = time.time()
    S = np.load(a.npy, mmap_mode="r")
    xyz = np.asarray(S[:, :3], np.float64)
    normales = np.asarray(S[:, 6:9], np.float32)
    lab = np.load(a.etiquetas)
    clases = json.loads(Path(a.info).read_text(encoding="utf-8"))["clases"]
    C = len(clases)
    z2, z3 = np.load(a.fsem), np.load(a.fssl)
    inv2, inv3 = z2["inverse"], z3["inverse"]
    assert len(inv2) == len(inv3) == len(xyz) == len(lab)
    reg0 = regiones_rejilla(xyz, a.celda)
    keep = np.bincount(reg0) >= a.min_puntos
    remap = np.cumsum(keep) - 1
    ok = keep[reg0]
    idx = np.flatnonzero(ok)
    reg = remap[reg0[ok]]
    R = int(keep.sum())
    tam = np.bincount(reg, minlength=R).astype(np.float64)
    print(f"regiones {R:,}; puntos agrupados {len(idx):,} de {len(xyz):,}", flush=True)
    Fsem = medias_por_region(reg, R, z2["feat"], inv2[idx])
    Fssl = medias_por_region(reg, R, z3["feat"], inv3[idx])
    g, nz, cen = rasgos_region(xyz[idx], normales[idx], reg, R)
    lg = np.concatenate([z2["logits_lab"], z2["logits_unlab"]], 1).astype(np.float32)
    lg -= lg.max(1, keepdims=True)
    p = np.exp(lg)
    conf_vox = (p.max(1) / p.sum(1)).astype(np.float32)
    conf = medias_por_region(reg, R, conf_vox[:, None], inv2[idx])[:, 0]
    y = lab[idx]
    cnt = np.zeros((R, C), np.int64)
    m = y >= 0
    np.add.at(cnt, (reg[m], y[m]), 1)
    np.savez_compressed(ruta, Fsem=Fsem, Fssl=Fssl, g=g, tam=tam, cen=cen, conf=conf, cnt=cnt, clases=np.array(clases))
    print(f"cache -> {ruta} ({time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------- evaluacion sobre regiones
def evaluar(T, quitar):
    """T (K x C) puntos por grupo y clase; las clases de `quitar` no se evaluan. mIoU 1 a 1 y por mayoria + IoU por clase."""
    T = T.copy()
    T[:, quitar] = 0
    pres = np.flatnonzero(T.sum(0) > 0)
    real = T.sum(0)
    out = {}
    fila, col = linear_sum_assignment(-T[:, pres])
    asig = {"uno_a_uno": {int(f): int(pres[c]) for f, c in zip(fila, col)},
            "mayoria": {k: int(T[k].argmax()) for k in range(T.shape[0]) if T[k].sum() > 0}}
    for nombre, A in asig.items():
        inter, pred = np.zeros(T.shape[1]), np.zeros(T.shape[1])
        for k, c in A.items():
            pred[c] += T[k].sum()
            inter[c] += T[k, c]
        iou = np.where(real + pred - inter > 0, inter / np.maximum(real + pred - inter, 1), 0.0)
        out[nombre] = (float(iou[pres].mean()), iou)
    return out


def tabla(lab_reg, K, cnt):
    Y = np.zeros((len(lab_reg), K))
    Y[np.arange(len(lab_reg)), lab_reg] = 1
    return Y.T @ cnt


def ari_pond(a, b, tam):
    r = np.repeat(np.arange(len(a)), np.maximum(1, (tam / tam.min()).astype(int)))
    return adjusted_rand_score(a[r], b[r])


# ---------------------------------------------------------------- preprocesado y afinidades
def pre_fuente(F, tam, modo, dim):
    F = F.astype(np.float64)
    if modo == "raw":
        return F
    mu = (F * tam[:, None]).sum(0) / tam.sum()
    Fc = F - mu
    if modo == "center":
        return Fc
    U, s, Vt = np.linalg.svd(Fc, full_matrices=False)
    d = min(dim, int((s > 1e-8 * s[0]).sum()))
    return Fc @ Vt[:d].T / (s[:d] / np.sqrt(len(F)) + 1e-12)


def snf(Ss, k=20, mu=0.5, it=20):
    """Similarity Network Fusion de dos matrices de similitud coseno (R x R)."""
    R = Ss[0].shape[0]
    P, Sk = [], []
    for S in Ss:
        D = np.clip(1 - S, 0, None)
        np.fill_diagonal(D, np.inf)
        kn = np.sort(D, axis=1)[:, :k].mean(1)
        np.fill_diagonal(D, 0)
        eps = (kn[:, None] + kn[None] + D) / 3 + 1e-12
        W = np.exp(-(D ** 2) / (mu * eps))
        np.fill_diagonal(W, 0)
        Pm = W / (2 * W.sum(1, keepdims=True) + 1e-12)
        np.fill_diagonal(Pm, 0.5)
        # similitud local: solo los k vecinos mas cercanos
        vec = np.argpartition(-W, k, axis=1)[:, :k]
        filas = np.repeat(np.arange(R), k)
        Wk = sp.csr_matrix((W[filas, vec.ravel()], (filas, vec.ravel())), shape=(R, R))
        Wk = sp.diags(1 / (np.asarray(Wk.sum(1)).ravel() + 1e-12)) @ Wk
        P.append(Pm)
        Sk.append(Wk.tocsr())
    for _ in range(it):
        n0 = Sk[0] @ P[1] @ Sk[0].T
        n1 = Sk[1] @ P[0] @ Sk[1].T
        for i, M in enumerate((n0, n1)):
            M = np.asarray(M)
            M = M / (2 * M.sum(1, keepdims=True) + 1e-12)
            M = (M + M.T) / 2
            np.fill_diagonal(M, 0.5)
            P[i] = M
    return (P[0] + P[1]) / 2


def ajustar(S, ref):
    """Lleva la afinidad S a la misma media y desviacion que la de referencia (para mezclarla con la geometrica igual)."""
    return (S - S.mean()) / (S.std() + 1e-12) * ref.std() + ref.mean()


def m4_generico(Fdesc, Saff, Sg, tam, K, rng, a_geo=0.5, M=8, gamma=1.0, iters=15):
    """Igual que m4_regiones, pero la afinidad semantica Saff se inyecta (ya calculada) y Sg viene precalculada."""
    Sb = a_geo * Sg + (1 - a_geo) * Saff
    m = min(M, len(Fdesc) - 1)
    cand = np.argpartition(-Sb, m, axis=1)[:, : m + 1]
    val = np.take_along_axis(Sb, cand, 1)
    vec = np.take_along_axis(cand, np.argsort(-val, axis=1), 1)
    w = np.take_along_axis(np.clip(Sb, 0, None), vec, 1)
    Fh = normalizar((w[..., None] * Fdesc[vec]).sum(1) / (w.sum(1, keepdims=True) + 1e-12))
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nombre", required=True, help="identifica la combinacion M2 + Sonata, p. ej. arch_original")
    ap.add_argument("--npy")
    ap.add_argument("--fsem")
    ap.add_argument("--fssl")
    ap.add_argument("--etiquetas")
    ap.add_argument("--info")
    ap.add_argument("--salida", required=True)
    ap.add_argument("--cache", help="ruta del cache de regiones (.npz); por defecto <salida>/cache_<nombre>.npz")
    ap.add_argument("--celda", type=float, default=0.75)
    ap.add_argument("--min-puntos", type=int, default=30)
    ap.add_argument("--K", type=int, nargs="+", default=[6, 8])
    ap.add_argument("--K-principal", type=int, default=8)
    ap.add_argument("--semillas", type=int, default=3)
    ap.add_argument("--alfas", type=float, nargs="+", default=[0.1, 0.25, 0.5, 0.75, 0.9])
    ap.add_argument("--a-geo", type=float, nargs="+", default=[0.0, 0.25, 0.5], help="pesos de la afinidad geometrica para el 2o paso")
    ap.add_argument("--ignorar", nargs="*", default=["otro"])
    ap.add_argument("--verificar", help="grupos_PL13.npz del M4 actual: comprueba que este banco reproduce su mIoU")
    a = ap.parse_args()

    sal = Path(a.salida) / a.nombre
    sal.mkdir(parents=True, exist_ok=True)
    cache = Path(a.cache) if a.cache else sal / "cache_regiones.npz"
    if not cache.exists():
        assert a.npy and a.fsem and a.fssl and a.etiquetas and a.info, "faltan entradas para construir el cache"
        construir_cache(a, cache)
    z = np.load(cache)
    Fsem, Fssl, g, tam, cen, conf, cnt = (z[k] for k in ("Fsem", "Fssl", "g", "tam", "cen", "conf", "cnt"))
    clases = [str(c) for c in z["clases"]]
    quitar = [clases.index(c) for c in a.ignorar if c in clases]
    R = len(tam)
    print(f"{a.nombre}: {R:,} regiones; clases {clases}; ignoradas {a.ignorar}", flush=True)

    # mitades espaciales A/B: se parte por la mediana (ponderada por puntos) del eje mas largo
    eje = int(np.argmax(cen.max(0) - cen.min(0)))
    orden = np.argsort(cen[:, eje])
    mitad = np.zeros(R, bool)
    mitad[orden[np.searchsorted(np.cumsum(tam[orden]), tam.sum() / 2):]] = True      # True = B
    mA, mB = ~mitad, mitad
    print(f"mitades por el eje {'xyz'[eje]}: A {mA.sum()} regiones, B {mB.sum()}", flush=True)

    # geometria (una vez)
    gz = (g - g.mean(0)) / (g.std(0) + 1e-9)
    Sg = np.exp(-((gz[:, None] - gz[None]) ** 2).sum(-1) / gz.shape[1]).astype(np.float32)
    rank_conf = (np.argsort(np.argsort(conf)) / (R - 1)).astype(np.float32)          # confianza de M2 como percentil 0..1

    # fuentes preprocesadas y sus cosenos
    PRE = {}
    for pre in ("raw", "center", "pca"):
        ns = normalizar(pre_fuente(Fsem, tam, pre, 32)).astype(np.float32)
        nl = normalizar(pre_fuente(Fssl, tam, pre, 64)).astype(np.float32)
        PRE[pre] = (ns, nl, ns @ ns.T, nl @ nl.T)
    ref_cos = 0.5 * PRE["raw"][2] + 0.5 * PRE["raw"][3]                                  # la afinidad del M4 actual
    SNF = {}

    def afinidad(pre, modo, alfa):
        ns, nl, cs, cl = PRE[pre]
        if modo == "lineal":
            return alfa * cs + (1 - alfa) * cl
        if modo == "prod":
            return ajustar(((1 + cs) / 2) ** alfa * ((1 + cl) / 2) ** (1 - alfa), ref_cos)
        if modo == "min":
            return ajustar(np.minimum(cs, cl), ref_cos)
        if modo == "conf":
            w = np.sqrt(np.outer(rank_conf, rank_conf))
            return w * cs + (1 - w) * cl
        if modo == "snf":
            if pre not in SNF:
                SNF[pre] = ajustar(snf([cs, cl]), ref_cos)
            return SNF[pre]
        raise ValueError(modo)

    def descriptor(pre, alfa):
        ns, nl = PRE[pre][0], PRE[pre][1]
        return np.hstack([np.sqrt(alfa) * ns, np.sqrt(1 - alfa) * nl]).astype(np.float32)

    def correr(cfg, K):
        S = afinidad(cfg["pre"], cfg["modo"], cfg["a"])
        Fd = descriptor(cfg["pre"], cfg["a"] if cfg["modo"] in ("lineal",) else 0.5)
        labs = [m4_generico(Fd, S, Sg, tam, K, np.random.default_rng(s), a_geo=cfg["a_geo"]) for s in range(a.semillas)]
        res = {"full": [], "A": [], "B": [], "iou": []}
        for l in labs:
            T = tabla(l, K, cnt)
            ev = evaluar(T, quitar)
            res["full"].append(ev["uno_a_uno"][0])
            res["iou"].append(ev["uno_a_uno"][1])
            res["A"].append(evaluar(tabla_sel(l, K, cnt, mA), quitar)["uno_a_uno"][0])
            res["B"].append(evaluar(tabla_sel(l, K, cnt, mB), quitar)["uno_a_uno"][0])
        ari = np.mean([ari_pond(labs[i], labs[j], tam) for i in range(len(labs)) for j in range(i + 1, len(labs))]) if len(labs) > 1 else 1.0
        return {"mIoU_full": float(np.mean(res["full"])), "std_full": float(np.std(res["full"])),
                "mIoU_A": float(np.mean(res["A"])), "mIoU_B": float(np.mean(res["B"])),
                "mIoU_may_full": None, "estabilidad_ari": float(ari),
                "iou_por_clase": {c: float(np.mean([x[i] for x in res["iou"]])) for i, c in enumerate(clases) if i not in quitar}}

    def tabla_sel(l, K, cnt_, m):
        Y = np.zeros((len(l), K))
        Y[np.arange(len(l)), l] = 1
        return Y[m].T @ cnt_[m]

    # ---- verificacion contra el M4 actual ----
    if a.verificar:
        zg = np.load(a.verificar)
        for K in a.K:
            clave = f"K{K}_M4"
            if clave in zg.files and len(zg[clave]) == R:
                v = evaluar(tabla(zg[clave].astype(int), K, cnt), quitar)["uno_a_uno"][0]
                print(f"verificacion K={K}: mIoU 1 a 1 del M4 actual recalculado aqui = {100 * v:.1f}", flush=True)
            else:
                print(f"verificacion K={K}: no hay {clave} con {R} regiones en {a.verificar}")

    # ---- configuraciones ----
    cfgs = [{"id": "base_M4_actual", "pre": "raw", "modo": "lineal", "a": 0.5, "a_geo": 0.5}]
    for pre in ("raw", "center", "pca"):
        for modo in ("lineal", "prod"):
            for al in a.alfas:
                cfgs.append({"id": f"{pre}|{modo}|a={al}", "pre": pre, "modo": modo, "a": al, "a_geo": 0.5})
        for modo in ("min", "conf", "snf"):
            cfgs.append({"id": f"{pre}|{modo}", "pre": pre, "modo": modo, "a": 0.5, "a_geo": 0.5})
    vistos, unicas = set(), []
    for c in cfgs:
        k = (c["pre"], c["modo"], c["a"], c["a_geo"])
        if k not in vistos:
            vistos.add(k)
            unicas.append(c)
    cfgs = unicas
    print(f"{len(cfgs)} configuraciones x K {a.K} x {a.semillas} semillas", flush=True)

    # referencia con el codigo original (m4_regiones) para comprobar el banco
    Fcat = np.hstack([normalizar(Fsem), normalizar(Fssl)]).astype(np.float32)
    ref = []
    for s in range(a.semillas):
        l = m4_regiones(Fcat, g, tam, a.K_principal, np.random.default_rng(s))
        ref.append(evaluar(tabla(l, a.K_principal, cnt), quitar)["uno_a_uno"][0])
    print(f"codigo original m4_regiones (media de {a.semillas} semillas), K={a.K_principal}: mIoU 1 a 1 = {100 * np.mean(ref):.1f}", flush=True)

    resultados = []
    t0 = time.time()
    for i, c in enumerate(cfgs, 1):
        for K in a.K:
            r = correr(c, K)
            resultados.append({**c, "K": K, **r})
        r8 = next(x for x in resultados[::-1] if x["id"] == c["id"] and x["K"] == a.K_principal) if a.K_principal in a.K else resultados[-1]
        print(f"[{i}/{len(cfgs)}] {c['id']:<22} K={a.K_principal}: {100 * r8['mIoU_full']:.1f} (A {100 * r8['mIoU_A']:.1f} | B {100 * r8['mIoU_B']:.1f})  "
              f"estab {r8['estabilidad_ari']:.2f}  [{time.time() - t0:.0f}s]", flush=True)

    # ---- 2o paso: mejores configuraciones de cada familia x peso de la geometria ----
    Kp = a.K_principal
    base = [r for r in resultados if r["id"] == "base_M4_actual" and r["K"] == Kp][0]
    familias = {}
    for r in resultados:
        if r["K"] == Kp and r["id"] != "base_M4_actual":
            familias.setdefault((r["pre"], r["modo"]), []).append(r)
    mejores = sorted((max(v, key=lambda r: r["mIoU_full"]) for v in familias.values()), key=lambda r: -r["mIoU_full"])[:4]
    for c in [{k: m[k] for k in ("id", "pre", "modo", "a", "a_geo")} for m in mejores] + [cfgs[0]]:
        for ag in a.a_geo:
            if ag == c["a_geo"]:
                continue
            cc = {**c, "a_geo": ag, "id": f"{c['id']}|geo={ag}"}
            for K in a.K:
                resultados.append({**cc, "K": K, **correr(cc, K)})
            r8 = [x for x in resultados if x["id"] == cc["id"] and x["K"] == Kp][0]
            print(f"  geo  {cc['id']:<30} K={Kp}: {100 * r8['mIoU_full']:.1f} (A {100 * r8['mIoU_A']:.1f} | B {100 * r8['mIoU_B']:.1f})", flush=True)

    # ---- protocolo honesto: elegir en una mitad, reportar en la otra ----
    cand = [r for r in resultados if r["K"] == Kp]
    elA = max(cand, key=lambda r: r["mIoU_A"])
    elB = max(cand, key=lambda r: r["mIoU_B"])
    honesto = {"elegida_en_A": {"id": elA["id"], "mIoU_A": elA["mIoU_A"], "mIoU_B_reportada": elA["mIoU_B"]},
               "elegida_en_B": {"id": elB["id"], "mIoU_B": elB["mIoU_B"], "mIoU_A_reportada": elB["mIoU_A"]},
               "estimacion_honesta": (elA["mIoU_B"] + elB["mIoU_A"]) / 2,
               "base_A": base["mIoU_A"], "base_B": base["mIoU_B"], "base_media_AB": (base["mIoU_A"] + base["mIoU_B"]) / 2,
               "base_full": base["mIoU_full"]}
    honesto["mejora_honesta_puntos"] = 100 * (honesto["estimacion_honesta"] - honesto["base_media_AB"])
    mejor_full = max(cand, key=lambda r: r["mIoU_full"])

    (sal / "resultados.json").write_text(json.dumps({"nombre": a.nombre, "K_principal": Kp, "semillas": a.semillas, "regiones": R,
                                                    "referencia_codigo_original": float(np.mean(ref)), "honesto": honesto,
                                                    "mejor_por_mIoU_total": mejor_full["id"], "resultados": resultados},
                                                   indent=1, ensure_ascii=False), encoding="utf-8")
    with open(sal / "resultados.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "pre", "modo", "a", "a_geo", "K", "mIoU_full", "std_semillas", "mIoU_A", "mIoU_B", "estabilidad_ari"] + [f"IoU_{c}" for c in clases if clases.index(c) not in quitar])
        for r in resultados:
            w.writerow([r["id"], r["pre"], r["modo"], r["a"], r["a_geo"], r["K"], round(100 * r["mIoU_full"], 2), round(100 * r["std_full"], 2),
                        round(100 * r["mIoU_A"], 2), round(100 * r["mIoU_B"], 2), round(r["estabilidad_ari"], 3)] +
                       [round(100 * r["iou_por_clase"][c], 1) for c in clases if clases.index(c) not in quitar])
    print("\n=== PROTOCOLO HONESTO (K=%d, mIoU 1 a 1, sin %s) ===" % (Kp, ",".join(a.ignorar)))
    print(f"base (M4 actual): A {100 * base['mIoU_A']:.1f} | B {100 * base['mIoU_B']:.1f} | media {100 * honesto['base_media_AB']:.1f} | escena completa {100 * base['mIoU_full']:.1f}")
    print(f"elegida en A: {elA['id']} -> en B {100 * elA['mIoU_B']:.1f}  |  elegida en B: {elB['id']} -> en A {100 * elB['mIoU_A']:.1f}")
    print(f"estimacion honesta: {100 * honesto['estimacion_honesta']:.1f}  (mejora sobre la base: {honesto['mejora_honesta_puntos']:+.1f} puntos)")
    print(f"mejor por mIoU de la escena completa (optimista): {mejor_full['id']} = {100 * mejor_full['mIoU_full']:.1f}")


if __name__ == "__main__":
    main()
