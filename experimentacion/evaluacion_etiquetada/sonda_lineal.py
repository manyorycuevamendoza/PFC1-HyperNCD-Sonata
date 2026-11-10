"""Sonda lineal (linear probing) y sonda k-NN sobre f_ssl congelado: el protocolo de evaluacion del paper de Sonata.

El paper de Sonata mide la calidad de la representacion entrenando SOLO una capa lineal con etiquetas sobre las features
congeladas y reportando mIoU, mAcc y allAcc. Aqui se hace lo mismo con la escena etiquetada, con dos cuidados:

  * DIVISION ESPACIAL. Train y test salen de bloques distintos (--bloque metros). Si se mezclaran puntos vecinos, el
    clasificador memorizaria el vecindario y el numero no diria nada de la representacion.
  * 2 pliegues: se entrena en la mitad A y se prueba en la B, y al reves. Se reporta el promedio.

Modos: "todas" (todas las clases) y "sin_otro" (se quitan los puntos de `otro` del entrenamiento y la prueba).
Sondas: "lineal" (regresion logistica multinomial, como la capa lineal del paper), "lineal_balanceada" (pesos por clase,
porque `otro` es el 61 %) y "knn" (k vecinos, coseno).

No usa regiones ni grupos: mide el descriptor punto a punto, asi que separa "¿el descriptor sirve?" de "¿M4 lo aprovecha?".

Uso:
    python sonda_lineal.py --feats f_ssl.npz --npy escena.npy --etiquetas escena_etiquetas.npy --info escena_etiquetas.json \
        --salida resultados/sonda_original --nombre original
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metricas import desde_predicciones  # noqa: E402


def pliegues(xy, bloque, rng):
    """Devuelve (idx_A, idx_B): puntos de dos mitades de bloques espaciales."""
    b = np.floor(xy / bloque).astype(np.int64)
    clave = b[:, 0] * 100003 + b[:, 1]
    _, bid = np.unique(clave, return_inverse=True)
    perm = rng.permutation(bid.max() + 1)
    mitad = perm[bid] % 2
    return np.flatnonzero(mitad == 0), np.flatnonzero(mitad == 1)


def entrenar_y_probar(Xtr, ytr, Xte, yte, tipo, clases_n, semilla):
    media, desv = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - media) / desv, (Xte - media) / desv
    if tipo == "knn":
        m = KNeighborsClassifier(n_neighbors=20, metric="cosine", algorithm="brute", n_jobs=-1).fit(Xtr, ytr)
    else:
        m = LogisticRegression(max_iter=150, C=1.0, class_weight="balanced" if tipo == "lineal_balanceada" else None,
                               random_state=semilla).fit(Xtr, ytr)
    return m.predict(Xte)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feats", required=True, help=".npz de extraer_features.py (feat, inverse)")
    ap.add_argument("--npy", required=True)
    ap.add_argument("--etiquetas", required=True)
    ap.add_argument("--info", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--nombre", default="f_ssl")
    ap.add_argument("--bloque", type=float, default=4.0, help="lado (m) de los bloques espaciales")
    ap.add_argument("--n-train", type=int, default=120_000)
    ap.add_argument("--n-test", type=int, default=60_000)
    ap.add_argument("--n-knn-train", type=int, default=60_000)
    ap.add_argument("--semilla", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.semilla)
    sal = Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)

    info = json.loads(Path(a.info).read_text(encoding="utf-8"))
    clases = info["clases"]
    lab = np.load(a.etiquetas)
    S = np.load(a.npy, mmap_mode="r")
    xy = np.asarray(S[:, :2], np.float64)
    z = np.load(a.feats)
    feat, inv = z["feat"], z["inverse"]
    print(f"[{a.nombre}] {len(lab):,} puntos; descriptor de {feat.shape[1]} dimensiones", flush=True)
    A, B = pliegues(xy, a.bloque, rng)
    resultado = {"nombre": a.nombre, "bloque_m": a.bloque, "dim": int(feat.shape[1]), "clases": clases, "modos": {}}

    for modo in ("todas", "sin_otro"):
        if modo == "sin_otro":
            cl = [c for c in clases if c != "otro"]
            remap = {clases.index(c): i for i, c in enumerate(cl)}
            y_all = np.array([remap.get(int(v), -1) for v in lab])
        else:
            cl, y_all = clases, lab.astype(int)
        validos = y_all >= 0
        resultado["modos"][modo] = {}
        for tipo in ("lineal", "lineal_balanceada", "knn"):
            t0 = time.time()
            pred_todo, y_todo = [], []
            for tr_idx, te_idx in ((A, B), (B, A)):
                tr = tr_idx[validos[tr_idx]]
                te = te_idx[validos[te_idx]]
                ntr = a.n_knn_train if tipo == "knn" else a.n_train
                tr = rng.choice(tr, min(len(tr), ntr), replace=False)
                te = rng.choice(te, min(len(te), a.n_test), replace=False)
                Xtr = feat[inv[tr]].astype(np.float32)
                Xte = feat[inv[te]].astype(np.float32)
                p = entrenar_y_probar(Xtr, y_all[tr], Xte, y_all[te], tipo, len(cl), a.semilla)
                pred_todo.append(p); y_todo.append(y_all[te])
            pred, y = np.concatenate(pred_todo), np.concatenate(y_todo)
            m = desde_predicciones(pred, y, cl)
            m["segundos"] = round(time.time() - t0, 1)
            m["puntos_prueba"] = int(len(y))
            resultado["modos"][modo][tipo] = m
            print(f"  {modo:9s} {tipo:18s} mIoU {100 * m['mIoU']:5.1f} | mAcc {100 * m['mAcc']:5.1f} | allAcc {100 * m['allAcc']:5.1f}  ({m['segundos']} s)", flush=True)

    (sal / f"sonda_{a.nombre}.json").write_text(json.dumps(resultado, indent=1, ensure_ascii=False), encoding="utf-8")
    fm = lambda x: "-" if x is None else f"{100 * x:.1f}"
    md = [f"# Sonda lineal de Sonata · `{a.nombre}`", "",
          f"Protocolo del paper de Sonata (clasificador lineal sobre `f_ssl` congelado), con **división espacial** en bloques de {a.bloque:g} m "
          "y 2 pliegues (entrena en la mitad A y prueba en la B, y al revés). Valores en %.", ""]
    for modo in resultado["modos"]:
        md += [f"## {'Todas las clases' if modo == 'todas' else 'Sin la clase `otro`'}", "", "| Sonda | mIoU | mAcc | allAcc |", "|---|---|---|---|"]
        for tipo, m in resultado["modos"][modo].items():
            md.append(f"| {tipo} | {fm(m['mIoU'])} | {fm(m['mAcc'])} | {fm(m['allAcc'])} |")
        md += ["", "Por clase (sonda `lineal_balanceada`):", "", "| Clase | Reales | TP | FP | FN | IoU | Precisión | Recall |", "|---|---|---|---|---|---|---|---|"]
        for c, d in resultado["modos"][modo]["lineal_balanceada"]["por_clase"].items():
            if d["puntos_reales"]:
                md.append(f"| {c} | {d['puntos_reales']:,} | {d['TP']:,} | {d['FP']:,} | {d['FN']:,} | {fm(d['IoU'])} | {fm(d['precision'])} | {fm(d['recall'])} |")
        md.append("")
    (sal / f"sonda_{a.nombre}.md").write_text("\n".join(md), encoding="utf-8")
    print("Listo ->", sal)


if __name__ == "__main__":
    main()
