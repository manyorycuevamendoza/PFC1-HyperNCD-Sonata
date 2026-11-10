"""Experimento: ¿el colapso de M2 (HyperNCD) es un problema de DOMINIO?

Pasa por M2, con el MISMO checkpoint y el MISMO codigo de inferencia (`predecir_hyperncd.py`), escenas de distintos dominios:
  * SemanticPOSS (el dominio donde se entreno)  -> control positivo: si aqui tambien colapsa, NO es dominio.
  * la capilla etiquetada y la Catedral          -> dominio nuevo.
  * la capilla submuestreada a la densidad de un LiDAR (--submuestreo) -> separa "dominio" de "densidad de puntos".

Metricas (no dependen de a que clase corresponde cada cabeza, asi son comparables entre dominios):
  * cabeza dominante (% de puntos), cabezas usadas (> 1 %), entropia normalizada de las cabezas, % marcado novedoso
  * rango efectivo de f_sem (exp de la entropia de los autovalores; 1 = todo en una direccion, ~100 = sano)
  * con etiquetas: NMI, ARI y mIoU 1 a 1 (emparejamiento hungaro cabeza -> clase) entre la cabeza y la clase real

Modos de entrada (se repiten las veces que haga falta):
  --semposs nombre=<carpeta de la secuencia con velodyne/ y labels/>   (usa --n-scans escaneos repartidos en la secuencia)
  --npy     nombre=<nube.npy de 9 columnas>[,<etiquetas.npy>[,<etiquetas.json>]]   (corre el modelo; necesita GPU)
  --fsem    nombre=<fsem.npz ya calculado>[,<etiquetas.npy>[,<etiquetas.json>]]   (no corre el modelo)

Uso (en Colab, con el entorno hyperncd-env):
    python medir_m2.py --ckpt <.ckpt> --salida resultados/epoca8 \
        --semposs semposs_val=.../sequences/03 --semposs semposs_train=.../sequences/00 \
        --npy capilla=.../capilla_etiquetada.npy,.../capilla_etiquetada_etiquetas.npy,.../capilla_etiquetada_etiquetas.json --submuestreo 80000
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
NCAB = 13          # cabezas totales (conocidas + nuevas); se ajusta con fijar_cabezas() segun el modelo
NL = 9             # cabezas conocidas


def fijar_cabezas(n_conocidas, n_nuevas):
    global NCAB, NL
    NL, NCAB = int(n_conocidas), int(n_conocidas) + int(n_nuevas)


# ------------------------------------------------------------------ metricas
def rango_efectivo(f, n=100_000, semilla=0):
    rng = np.random.default_rng(semilla)
    x = np.asarray(f[rng.choice(len(f), min(len(f), n), replace=False)], np.float32)
    _, s, _ = np.linalg.svd(x - x.mean(0), full_matrices=False)
    v = s ** 2 / (s ** 2).sum()
    p = v[v > 0]
    return float(np.exp(-(p * np.log(p)).sum())), float(v[:3].sum())


def _H(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def desde_cruce(T):
    """T[k, c] = puntos de la cabeza k con clase real c. Devuelve NMI, ARI y mIoU 1 a 1 (hungaro)."""
    from scipy.optimize import linear_sum_assignment
    T = np.asarray(T, np.float64)
    n = T.sum()
    if n == 0:
        return dict(NMI=None, ARI=None, mIoU=None, IoU_por_clase=[])
    pk, pc = T.sum(1) / n, T.sum(0) / n
    pkc = T / n
    mask = pkc > 0
    mi = float((pkc[mask] * np.log(pkc[mask] / (pk[:, None] * pc[None, :])[mask])).sum())
    hk, hc = _H(pk), _H(pc)
    nmi = mi / (0.5 * (hk + hc)) if (hk + hc) > 0 else 1.0
    comb2 = lambda x: x * (x - 1) / 2.0
    s_ij, s_i, s_j = comb2(T).sum(), comb2(T.sum(1)).sum(), comb2(T.sum(0)).sum()
    esp = s_i * s_j / comb2(n)
    mx = 0.5 * (s_i + s_j)
    ari = (s_ij - esp) / (mx - esp) if mx != esp else 1.0
    fila, col = linear_sum_assignment(-T)
    iou = np.zeros(T.shape[1])
    for k, c in zip(fila, col):
        tp = T[k, c]
        fp, fn = T[k].sum() - tp, T[:, c].sum() - tp
        iou[c] = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return dict(NMI=float(nmi), ARI=float(ari), mIoU=float(iou.mean()), IoU_por_clase=iou.tolist())


def resumen_cabezas(h, nl=None):
    nl = NL if nl is None else nl
    c = np.bincount(h, minlength=NCAB).astype(np.float64)
    p = c / c.sum()
    return dict(cabeza_dominante=int(c.argmax()), pct_dominante=float(100 * p.max()),
                cabezas_usadas_mas_1pct=int((p > 0.01).sum()), entropia_normalizada=float(_H(p) / np.log(NCAB)),
                pct_novedoso=float(100 * p[nl:].sum()), distribucion_pct=(100 * p).tolist())


def cruce(h, y, ncl):
    ok = y >= 0
    T = np.zeros((NCAB, ncl), np.int64)
    np.add.at(T, (h[ok], y[ok]), 1)
    return T


# ------------------------------------------------------------------ entrada / modelo
_modelo = {}


def correr_modelo(ckpt, coord, voxel):
    """Mismo codigo que predecir_hyperncd.py: cuantiza con ME, features = 1, forward, argmax de [conocidas ; novedosas]."""
    import torch
    sys.path.insert(0, str(RAIZ))
    import MinkowskiEngine as ME
    from predecir_hyperncd import cargar_modelo_hyperncd, cuantizar_coordenadas
    if ckpt not in _modelo:
        _modelo[ckpt] = cargar_modelo_hyperncd(ckpt)
    modelo, info = _modelo[ckpt]
    fijar_cabezas(info["num_labeled"], info["num_unlabeled"])
    dc, um, inv = cuantizar_coordenadas(np.ascontiguousarray(coord, np.float32), voxel)
    with torch.no_grad():
        bc = ME.utils.batched_coordinates([dc]).int().cuda()
        sp = ME.SparseTensor(features=torch.ones((len(dc), 1), dtype=torch.float32, device="cuda"), coordinates=bc)
        out = modelo(sp)
        torch.cuda.synchronize()
    f = out["feats"].detach().cpu().numpy().astype(np.float32)
    lg = np.concatenate([out["logits_lab"].detach().cpu().numpy(), out["logits_unlab"].detach().cpu().numpy()], 1)
    return f, lg.argmax(1).astype(np.int32), inv, info


def leer_semposs(carpeta, n_scans):
    import yaml
    cfg = yaml.safe_load(open(RAIZ / "hyperncd/config/semposs_dataset.yaml", encoding="utf-8"))
    mapa = {int(k): int(v) for k, v in cfg["learning_map"].items()}
    lut = np.full(max(mapa) + 1, -1, np.int64)
    for k, v in mapa.items():
        lut[k] = v
    carpeta = Path(carpeta)
    bins = sorted((carpeta / "velodyne").glob("*.bin"))
    assert bins, f"No hay .bin en {carpeta}/velodyne"
    sel = np.unique(np.linspace(0, len(bins) - 1, min(n_scans, len(bins))).round().astype(int))
    for i in sel:
        b = bins[i]
        xyz = np.fromfile(b, dtype=np.float32).reshape(-1, 4)[:, :3]
        lab = np.fromfile(carpeta / "labels" / (b.stem + ".label"), dtype=np.int32) & 0xFFFF
        yield b.stem, xyz, lut[np.clip(lab, 0, len(lut) - 1)]


def evaluar_escena(nombre, h, feat, y, clases, ignorar=None):
    """h: cabeza por punto. y: clase por punto (-1 = sin etiqueta) o None."""
    r = dict(nombre=nombre, n_puntos=int(len(h)), **resumen_cabezas(h))
    r["rango_efectivo"], r["varianza_3_componentes"] = rango_efectivo(feat)
    if y is not None:
        for modo in ("todas", "sin_otro"):
            if modo == "sin_otro" and (ignorar is None or ignorar not in clases):
                continue
            yy = y.copy()
            if modo == "sin_otro":
                yy[yy == clases.index(ignorar)] = -1
            T = cruce(h, yy, len(clases))
            r[modo] = dict(**desde_cruce(T), cruce_cabeza_x_clase=T.tolist(), puntos_evaluados=int(T.sum()))
    return r


# ------------------------------------------------------------------ principal
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", help="checkpoint de HyperNCD (necesario si hay --semposs o --npy)")
    ap.add_argument("--salida", required=True)
    ap.add_argument("--semposs", action="append", default=[])
    ap.add_argument("--npy", action="append", default=[])
    ap.add_argument("--fsem", action="append", default=[])
    ap.add_argument("--n-scans", type=int, default=8)
    ap.add_argument("--voxel", type=float, default=0.05)
    ap.add_argument("--submuestreo", type=int, default=0, help="ademas de cada --npy, una variante con este numero de puntos al azar")
    ap.add_argument("--etiqueta", default="", help="texto que identifica la corrida (p. ej. epoca8)")
    a = ap.parse_args()
    sal = Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)
    resultados, clases_por_escena, T_semposs = [], {}, {}
    cfg_clases = ["person", "rider", "car", "trunk", "plants", "traffic-sign", "pole", "trashcan", "building", "cone/stone", "fence", "bike", "ground"]

    def spec(s):
        nombre, rest = s.split("=", 1)
        partes = rest.split(",")
        return nombre, partes[0], (partes[1] if len(partes) > 1 else None), (partes[2] if len(partes) > 2 else None)

    for s in a.semposs:                                   # ---- SemanticPOSS: varios escaneos, media y desviacion
        nombre, carpeta, _, _ = spec(s)
        por_scan, Tacum = [], None
        for stem, xyz, y in leer_semposs(carpeta, a.n_scans):
            f, pl, inv, info = correr_modelo(a.ckpt, xyz, a.voxel)
            r = evaluar_escena(f"{nombre}/{stem}", pl[inv], f, y, cfg_clases)
            if Tacum is None:
                Tacum = np.zeros((NCAB, 13), np.int64)
            Tacum += np.array(r["todas"]["cruce_cabeza_x_clase"], np.int64)
            por_scan.append(r)
            print(f"  {nombre}/{stem}: dominante {r['pct_dominante']:.1f} % | rango {r['rango_efectivo']:.1f} | NMI {r['todas']['NMI']:.3f}", flush=True)
        agg = dict(nombre=nombre, n_escaneos=len(por_scan), escaneos=[p["nombre"] for p in por_scan])
        for k in ("pct_dominante", "cabezas_usadas_mas_1pct", "entropia_normalizada", "pct_novedoso", "rango_efectivo"):
            v = np.array([p[k] for p in por_scan], float)
            agg[k], agg[k + "_std"] = float(v.mean()), float(v.std())
        for k in ("NMI", "ARI", "mIoU"):
            v = np.array([p["todas"][k] for p in por_scan], float)
            agg[k], agg[k + "_std"] = float(v.mean()), float(v.std())
        agg["distribucion_pct"] = np.mean([p["distribucion_pct"] for p in por_scan], 0).tolist()
        agg["todas"] = dict(NMI=agg["NMI"], ARI=agg["ARI"], mIoU=agg["mIoU"], cruce_cabeza_x_clase=Tacum.tolist())
        agg["tipo"] = "semposs"
        clases_por_escena[nombre] = cfg_clases
        resultados.append(agg)

    def cargar_y(rutas_y, rutas_json, n):
        if not rutas_y:
            return None, None, None
        y = np.load(rutas_y).astype(np.int64)
        info = json.loads(Path(rutas_json).read_text(encoding="utf-8")) if rutas_json else None
        cl = info["clases"] if info else [str(i) for i in range(int(y.max()) + 1)]
        assert len(y) == n, f"etiquetas ({len(y)}) y nube ({n}) no coinciden"
        return y, cl, ("otro" if "otro" in cl else None)

    for s in a.npy:                                       # ---- nube propia: corre el modelo
        nombre, ruta, ry, rj = spec(s)
        S = np.load(ruta, mmap_mode="r")
        coord = np.asarray(S[:, :3], np.float32)
        y, cl, ign = cargar_y(ry, rj, len(coord))
        variantes = [(nombre, np.arange(len(coord)))]
        if a.submuestreo and a.submuestreo < len(coord):
            variantes.append((f"{nombre}_{a.submuestreo // 1000}k", np.sort(np.random.default_rng(0).choice(len(coord), a.submuestreo, replace=False))))
        for nom, idx in variantes:
            t0 = time.time()
            f, pl, inv, info = correr_modelo(a.ckpt, coord[idx], a.voxel)
            r = evaluar_escena(nom, pl[inv], f, None if y is None else y[idx], cl, ign)
            r["tipo"] = "npy"; r["voxeles"] = int(len(f)); r["segundos"] = round(time.time() - t0, 1)
            if cl: clases_por_escena[nom] = cl
            resultados.append(r)
            print(f"  {nom}: {len(idx):,} pts -> {len(f):,} voxeles | dominante {r['pct_dominante']:.1f} % | rango {r['rango_efectivo']:.1f}"
                  + (f" | NMI {r['todas']['NMI']:.3f}" if "todas" in r else ""), flush=True)

    for s in a.fsem:                                      # ---- f_sem ya calculado
        nombre, ruta, ry, rj = spec(s)
        z = np.load(ruta)
        fijar_cabezas(z["logits_lab"].shape[1], z["logits_unlab"].shape[1])
        h = z["pred_label"][z["inverse"]].astype(np.int64)
        y, cl, ign = cargar_y(ry, rj, len(h))
        r = evaluar_escena(nombre, h, z["feat"], y, cl, ign)
        r["tipo"] = "fsem_guardado"
        if cl: clases_por_escena[nombre] = cl
        resultados.append(r)
        print(f"  {nombre}: dominante {r['pct_dominante']:.1f} % | rango {r['rango_efectivo']:.1f}" + (f" | NMI {r['todas']['NMI']:.3f}" if "todas" in r else ""), flush=True)

    (sal / "resultados.json").write_text(json.dumps(dict(etiqueta=a.etiqueta, ckpt=a.ckpt, voxel=a.voxel, escenas=resultados,
                                                         clases=clases_por_escena), indent=1, ensure_ascii=False), encoding="utf-8")
    f3 = lambda x: "-" if x is None else f"{x:.3f}"
    md = [f"# M2: ¿colapsa por dominio? · {a.etiqueta}", "", f"Checkpoint: `{a.ckpt}` · voxel {a.voxel} m", "",
          "| Escena | Puntos | Cabeza dominante (%) | Cabezas > 1 % | Rango efectivo f_sem | NMI | ARI | mIoU 1 a 1 |", "|---|---|---|---|---|---|---|---|"]
    for r in resultados:
        t = r.get("todas") or {}
        n = r.get("n_puntos", "-")
        md.append(f"| {r['nombre']} | {n if n == '-' else f'{n:,}'} | {r['pct_dominante']:.1f} | {r['cabezas_usadas_mas_1pct']:.0f} | {r['rango_efectivo']:.1f} | {f3(t.get('NMI'))} | {f3(t.get('ARI'))} | {f3(t.get('mIoU'))} |")
        if "sin_otro" in r:
            t = r["sin_otro"]
            md.append(f"| {r['nombre']} (sin «otro») | | | | | {f3(t['NMI'])} | {f3(t['ARI'])} | {f3(t['mIoU'])} |")
    (sal / "resumen.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    graficar(resultados, clases_por_escena, sal, a.etiqueta)
    print("\n".join(md))
    print("Listo ->", sal)


def graficar(resultados, clases_por_escena, sal, etiqueta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nombres = [r["nombre"] for r in resultados]
    col = ["#2b7bba" if r.get("tipo") == "semposs" else "#e07b39" for r in resultados]
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.2))
    for a_, (k, t) in zip(ax, [("pct_dominante", "% de puntos en la cabeza dominante\n(100 = colapso total)"), ("rango_efectivo", "rango efectivo de f_sem\n(1 = colapso, ~100 = sano)"),
                               ("entropia_normalizada", "entropía normalizada de las cabezas\n(0 = una sola, 1 = reparto parejo)"), ("NMI", "NMI cabeza vs clase real\n(0 = no separa, 1 = perfecto)")]):
        v = [(r.get(k) if k != "NMI" else (r.get("todas") or {}).get("NMI")) for r in resultados]
        sd = [r.get(k + "_std", 0) or 0 for r in resultados]
        a_.bar(range(len(v)), [x or 0 for x in v], yerr=sd, color=col)
        a_.set_xticks(range(len(v))); a_.set_xticklabels(nombres, rotation=60, ha="right", fontsize=8); a_.set_title(t, fontsize=9)
    fig.suptitle(f"M2 en distintos dominios · {etiqueta}  (azul = SemanticPOSS, dominio de entrenamiento; naranja = otros)", fontsize=10)
    fig.tight_layout(); fig.savefig(sal / "m2_dominio.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(1.0 + 0.9 * len(resultados), 4.2))
    base = np.zeros(len(resultados))
    cmap = plt.get_cmap("tab20")
    for k in range(NCAB):
        v = np.array([r["distribucion_pct"][k] for r in resultados])
        ax.bar(range(len(resultados)), v, bottom=base, color=cmap(k), label=f"cabeza {k}" + (" (nov.)" if k >= NL else ""))
        base += v
    ax.set_xticks(range(len(resultados))); ax.set_xticklabels(nombres, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("% de puntos"); ax.set_title(f"Reparto de puntos entre las {NCAB} cabezas · {etiqueta}")
    ax.legend(ncol=2, fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left"); fig.tight_layout(); fig.savefig(sal / "m2_cabezas.png", dpi=130); plt.close(fig)

    con_T = [r for r in resultados if "todas" in r and r["nombre"] in clases_por_escena]
    if con_T:
        fig, ax = plt.subplots(1, len(con_T), figsize=(5.2 * len(con_T), 4.6), squeeze=False)
        for a_, r in zip(ax[0], con_T):
            T = np.array(r["todas"]["cruce_cabeza_x_clase"], float)
            Tn = T / np.maximum(T.sum(0, keepdims=True), 1)           # cada columna = una clase real
            a_.imshow(Tn, aspect="auto", cmap="Blues", vmin=0, vmax=1)
            cl = clases_por_escena[r["nombre"]]
            a_.set_xticks(range(len(cl))); a_.set_xticklabels(cl, rotation=70, ha="right", fontsize=7)
            a_.set_yticks(range(NCAB)); a_.set_ylabel("cabeza predicha"); a_.set_title(f"{r['nombre']}\ncada columna: a qué cabezas va una clase real", fontsize=8)
        fig.tight_layout(); fig.savefig(sal / "m2_cabeza_x_clase.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
