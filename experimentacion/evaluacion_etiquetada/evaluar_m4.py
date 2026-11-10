"""Evalua con etiquetas los grupos del Modulo 4 (y de M2, M3, M4 base) sobre una escena etiquetada.

Entrada: uno o varios `grupos_PL13.npz` que escribe `modulo4_real/correr_modulo4_real.py` (uno por
variante de M3, por ejemplo original y adaptado) y las etiquetas por punto de
`preparar_etiquetas.py mapear`.

Los grupos no tienen nombre, asi que hay que decir a que clase corresponde cada uno. Se reportan DOS
formas (ninguna usa mas informacion que las etiquetas de evaluacion):

  - mayoria   cada grupo se asigna a la clase con mas puntos (muchos grupos pueden ir a la misma clase).
              Es la medida habitual en segmentacion no supervisada con K >= numero de clases.
  - 1 a 1     asignacion hungara: cada clase recibe a lo sumo un grupo. Mas estricta.

Metricas: mIoU, mIoU sin la clase de fondo ("otro"), exactitud, ARI y NMI contra las etiquetas, y la
cobertura (M4 solo agrupa los puntos que M2 marca como novedosos; el resto no se evalua).

Referencia: "todo a la clase mayoritaria". Si un mIoU no supera esa linea, no dice nada.

Uso:
    python experimentacion/evaluacion_etiquetada/evaluar_m4.py \
        --npy escena.npy --etiquetas escena_etiquetas.npy --info escena_etiquetas.json \
        --grupos original=resultados/m4_original/grupos_PL13.npz adaptado=resultados/m4_adaptado/grupos_PL13.npz \
        --salida resultados/evaluacion
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from metricas import desde_cruce, evaluar_T

VARIANTES = ["M2", "M3", "M4_base", "M4"]


def tabla_cruce(grupo, verdad, K, C):
    T = np.zeros((K, C), np.int64)
    np.add.at(T, (grupo, verdad), 1)
    return T


def iou_por_clase(T, asignacion, C):
    """T: grupos x clases. asignacion[k] = clase del grupo k (o -1). IoU de cada clase."""
    pred_clase = np.zeros(C)
    inter = np.zeros(C)
    for k, c in enumerate(asignacion):
        if c >= 0:
            pred_clase[c] += T[k].sum()
            inter[c] += T[k, c]
    real = T.sum(0)
    union = real + pred_clase - inter
    return np.where(union > 0, inter / np.maximum(union, 1), 0.0), inter.sum() / max(T.sum(), 1)


def metricas(grupo, verdad, K, clases):
    return evaluar_T(tabla_cruce(grupo, verdad, K, len(clases)), clases)


def escribir_salidas(resultado, sal, ignorar=()):
    """Escribe evaluacion.json, por_clase.csv, evaluacion.md y evaluacion_miou.png a partir del dict `resultado`."""
    sal = Path(sal)
    sal.mkdir(parents=True, exist_ok=True)
    ignorar = list(ignorar)
    clases = resultado["clases"]
    total = np.array([resultado["puntos_por_clase"][c] for c in clases])
    referencia = resultado["referencia_todo_a_la_mayoritaria"]
    (sal / "evaluacion.json").write_text(json.dumps(resultado, indent=1, ensure_ascii=False))

    # CSV largo: una fila por (variante de M3, K, modulo, asignacion, clase). Lo leen las tablas y el visor.
    cols = ["puntos_reales", "TP", "FP", "FN", "TN", "IoU", "precision", "recall", "F1", "FPR", "FNR"]
    filas = ["m3,K,modulo,asignacion,clase," + ",".join(cols)]
    for nombre, bloque in resultado["variantes"].items():
        for K, mods in bloque["K"].items():
            for var, m in mods.items():
                for asign in ("mayoria", "uno_a_uno"):
                    for c, d in m[asign]["por_clase"].items():
                        filas.append(f"{nombre},{K},{var},{asign},{c}," + ",".join("" if d[k] is None else f"{d[k]:.6g}" for k in cols))
    (sal / "por_clase.csv").write_text("\n".join(filas) + "\n", encoding="utf-8")

    # ---- tabla legible ----
    md = ["# Evaluación con etiquetas", "",
          f"Clases: {', '.join(f'{c} ({n:,})' for c, n in zip(clases, total))}",
          (f"**Clases ignoradas en la evaluación:** {', '.join(ignorar)}" if ignorar else ""), "",
          f"**Referencia (todo a '{referencia['clase']}'):** mIoU {referencia['mIoU']:.3f}, "
          f"exactitud {referencia['exactitud']:.3f}. Un mIoU que no supere esto no dice nada.", ""]
    for nombre, bloque in resultado["variantes"].items():
        md += [f"## M3 = {nombre}", "",
               f"Puntos evaluados: {bloque['puntos_evaluados']:,} ({bloque['cobertura_total']:.1%} de los etiquetados; "
               "solo los que M2 marca como novedosos).", "",
               "| K | Variante | mIoU (mayoría) | mIoU sin 'otro' | mAcc | allAcc | mIoU 1 a 1 | ARI | NMI |",
               "|---|---|---|---|---|---|---|---|---|"]
        for K in sorted(bloque["K"]):
            for var in VARIANTES:
                if var not in bloque["K"][K]:
                    continue
                m = bloque["K"][K][var]
                sf = m["mayoria"]["mIoU_sin_otro"]
                md.append(f"| {K} | {var} | {m['mayoria']['mIoU']:.3f} | {'-' if sf is None else f'{sf:.3f}'} | "
                          f"{m['mayoria']['mAcc']:.3f} | {m['mayoria']['allAcc']:.3f} | {m['uno_a_uno']['mIoU']:.3f} | {m['ARI']:.3f} | {m['NMI']:.3f} |")
        md.append("")
        # tabla por clase (con falsos positivos y negativos) para el K mas cercano al numero de clases
        Kc = min((K for K in bloque["K"] if K >= sum(1 for n in total if n > 0)), default=max(bloque["K"]))
        md += [f"### Por clase (K = {Kc}, asignación por mayoría): falsos positivos y negativos", "",
               "| Módulo | Clase | Puntos reales | TP | FP | FN | IoU | Precisión | Recall | F1 | FPR | FNR |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        fm = lambda x: "-" if x is None else f"{x:.3f}"
        for var in VARIANTES:
            if var not in bloque["K"][Kc]:
                continue
            for c, d in bloque["K"][Kc][var]["mayoria"]["por_clase"].items():
                if d["puntos_reales"] == 0:
                    continue
                md.append(f"| {var} | {c} | {d['puntos_reales']:,} | {d['TP']:,} | {d['FP']:,} | {d['FN']:,} | {fm(d['IoU'])} | "
                          f"{fm(d['precision'])} | {fm(d['recall'])} | {fm(d['F1'])} | {fm(d['FPR'])} | {fm(d['FNR'])} |")
        md.append("")
    (sal / "evaluacion.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))

    # ---- figura: mIoU por variante y K, una barra por M3 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nombres = list(resultado["variantes"])
    Ks = sorted({K for b in resultado["variantes"].values() for K in b["K"]})
    fig, ejes = plt.subplots(1, len(Ks), figsize=(5.2 * len(Ks), 4.6), squeeze=False)
    ancho = 0.8 / max(1, len(nombres))
    for j, K in enumerate(Ks):
        ax = ejes[0, j]
        for i, n in enumerate(nombres):
            vals = [resultado["variantes"][n]["K"].get(K, {}).get(v, {}).get("mayoria", {}).get("mIoU", 0) for v in VARIANTES]
            ax.bar(np.arange(len(VARIANTES)) + i * ancho, vals, ancho, label=f"M3 {n}")
        ax.axhline(referencia["mIoU"], color="k", ls="--", lw=1, label="todo a la mayoritaria")
        ax.set_xticks(np.arange(len(VARIANTES)) + ancho * (len(nombres) - 1) / 2)
        ax.set_xticklabels(VARIANTES)
        ax.set_title(f"K = {K}")
        ax.set_ylabel("mIoU (asignación por mayoría)")
        ax.set_ylim(0, 1)
        if j == 0:
            ax.legend(fontsize=8)
    fig.suptitle("Módulo 4 con etiquetas: mIoU por variante")
    fig.tight_layout()
    fig.savefig(sal / "evaluacion_miou.png", dpi=130)
    print("figura ->", sal / "evaluacion_miou.png")



def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--etiquetas", required=True, help="<escena>_etiquetas.npy (int8 por punto del .npy)")
    ap.add_argument("--info", required=True, help="<escena>_etiquetas.json (nombres de clases)")
    ap.add_argument("--grupos", nargs="+", required=True, help="nombre=ruta/grupos_PL13.npz (uno o mas)")
    ap.add_argument("--salida", required=True)
    ap.add_argument("--ignorar", nargs="*", default=[],
                    help="clases que NO se evaluan (por ejemplo: otro). Sus puntos se tratan como sin etiqueta")
    a = ap.parse_args()

    lab = np.load(a.etiquetas)
    info = json.loads(Path(a.info).read_text())
    clases = info["clases"]
    C = len(clases)
    for c in a.ignorar:
        assert c in clases, f"Clase '{c}' no existe. Clases: {clases}"
        lab[lab == clases.index(c)] = -1          # como si no tuvieran etiqueta: no entran a ninguna metrica
    sal = Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)

    total = np.bincount(lab[lab >= 0], minlength=C)
    mayoritaria = int(total.argmax())
    ref_iou = np.zeros(C)
    ref_iou[mayoritaria] = total[mayoritaria] / total.sum()
    referencia = {"clase": clases[mayoritaria], "mIoU": float(ref_iou[total > 0].mean()),
                  "exactitud": float(total[mayoritaria] / total.sum())}

    resultado = {"clases": clases, "puntos_por_clase": {c: int(n) for c, n in zip(clases, total)},
                 "referencia_todo_a_la_mayoritaria": referencia, "variantes": {}}
    for item in a.grupos:
        nombre, ruta = item.split("=", 1)
        z = np.load(ruta)
        idx, reg = z["idx_puntos"], z["region"]
        verdad_pt = lab[idx]
        ok = verdad_pt >= 0
        cobertura = {c: float(((lab[idx] == k).sum()) / max(total[k], 1)) for k, c in enumerate(clases)}
        bloque = {"puntos_evaluados": int(ok.sum()), "cobertura_total": float(ok.sum() / total.sum()),
                  "cobertura_por_clase": cobertura, "K": {}}
        for clave in z.files:
            if not clave.startswith("K"):
                continue
            K, var = clave[1:].split("_", 1)
            K = int(K)
            grupo_pt = z[clave][reg][ok]
            bloque["K"].setdefault(K, {})[var] = metricas(grupo_pt, verdad_pt[ok], int(grupo_pt.max()) + 1, clases)
        resultado["variantes"][nombre] = bloque
    escribir_salidas(resultado, sal, a.ignorar)

if __name__ == "__main__":
    main()
