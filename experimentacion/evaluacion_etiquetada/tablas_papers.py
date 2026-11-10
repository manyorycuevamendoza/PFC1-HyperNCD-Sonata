"""Tablas y graficos de resultados, con el formato de los papers de HyperNCD y de Sonata, mas FP/FN por clase.

Lee `evaluacion/todas/evaluacion.json` y `evaluacion/sin_otro/evaluacion.json` (de `reevaluar_desde_cruces.py` o de `evaluar_m4.py`).

Tablas
  estilo_hyperncd   IoU por clase tras emparejamiento hungaro (1 a 1), mas "Novel", "Known" y "All" como en las Tablas 1 y 2 del
                    paper de HyperNCD. En esta escena TODAS las clases son novedosas (el modelo no las vio etiquetadas), asi que
                    "Known" no existe y "All" coincide con "Novel".
  estilo_sonata     mIoU, mAcc y allAcc como en las tablas de segmentacion semantica del paper de Sonata. OJO: aqui son de un
                    AGRUPAMIENTO asignado por mayoria, NO del protocolo de Sonata (sonda lineal con etiquetas); ese esta en
                    `sonda_lineal.py`.
  fp_fn             falsos positivos y negativos por clase.
Graficos (en --salida/graficos): miou_por_modulo, iou_por_clase, fp_fn_por_clase, confusion.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODULOS = ["M2", "M3", "M4_base", "M4"]
COLOR = {"original": "#7a8ca5", "adaptado": "#e07b39"}


def cargar(ruta):
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


def get(d, variante, K, mod):
    return d["variantes"][variante]["K"][str(K)][mod]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evaluacion", required=True, help="carpeta con todas/ y sin_otro/")
    ap.add_argument("--salida", required=True)
    ap.add_argument("--k-ref", type=int, default=8, help="K de referencia para graficos por clase")
    a = ap.parse_args()
    ev = Path(a.evaluacion)
    sal = Path(a.salida)
    (sal / "tablas").mkdir(parents=True, exist_ok=True)
    (sal / "graficos").mkdir(parents=True, exist_ok=True)
    datos = {"todas": cargar(ev / "todas/evaluacion.json"), "sin_otro": cargar(ev / "sin_otro/evaluacion.json")}
    variantes = list(datos["todas"]["variantes"])
    Ks = sorted(int(k) for k in datos["todas"]["variantes"][variantes[0]]["K"])
    f3 = lambda x: "-" if x is None else f"{x * 100:.1f}"

    # ============================ estilo HyperNCD ============================
    md = ["# Tablas al estilo de HyperNCD (IoU %, emparejamiento húngaro 1 a 1)", "",
          "Como en las Tablas 1 y 2 del paper: IoU por clase tras emparejar grupos con clases, y la columna **Novel** (promedio de las novedosas). "
          "Aquí **todas las clases son novedosas** (el modelo no las vio con etiqueta), así que **Known** no existe y **All = Novel**. "
          "No hay fila 'Full' (modelo supervisado) porque no se entrenó uno con esta escena.", ""]
    filas_csv = [["evaluacion", "K", "m3", "modulo", "clase", "IoU"]]
    for modo, d in datos.items():
        clases = d["clases"]
        for K in Ks:
            md += [f"## {'Todas las clases' if modo == 'todas' else 'Sin la clase `otro`'} · K = {K}", "",
                   "| M3 | Método | " + " | ".join(clases) + " | Novel | Known | All |", "|---|---|" + "---|" * (len(clases) + 3)]
            for v in variantes:
                for m in MODULOS:
                    r = get(d, v, K, m)["uno_a_uno"]
                    ious = [r["iou_por_clase"][c] for c in clases]
                    md.append(f"| {v} | {m} | " + " | ".join(f3(x) for x in ious) + f" | {f3(r['mIoU'])} | - | {f3(r['mIoU'])} |")
                    for c, x in zip(clases, ious):
                        filas_csv.append([modo, K, v, m, c, f"{x:.6f}"])
            md.append("")
    (sal / "tablas/estilo_hyperncd.md").write_text("\n".join(md), encoding="utf-8")
    with open(sal / "tablas/estilo_hyperncd.csv", "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(filas_csv)

    # ============================ estilo Sonata ============================
    md = ["# Tablas al estilo de Sonata (mIoU, mAcc, allAcc, en %)", "",
          "Mismas tres columnas que las tablas de segmentación semántica del paper de Sonata. **Advertencia:** estos números son de un "
          "**agrupamiento sin supervisión** cuyos grupos se asignan a la clase mayoritaria; **no son** el *linear probing* del paper "
          "(clasificador lineal entrenado con etiquetas sobre `f_ssl` congelado). Ese protocolo está en `sonda_lineal.py`.", ""]
    filas_csv = [["evaluacion", "K", "m3", "modulo", "mIoU", "mAcc", "allAcc", "mIoU_1a1"]]
    for modo, d in datos.items():
        md += [f"## {'Todas las clases' if modo == 'todas' else 'Sin la clase `otro`'}", "",
               "| K | M3 | Método | mIoU | mAcc | allAcc | mIoU 1 a 1 |", "|---|---|---|---|---|---|---|"]
        for K in Ks:
            for v in variantes:
                for m in MODULOS:
                    r = get(d, v, K, m)
                    my, uno = r["mayoria"], r["uno_a_uno"]
                    md.append(f"| {K} | {v} | {m} | {f3(my['mIoU'])} | {f3(my['mAcc'])} | {f3(my['allAcc'])} | {f3(uno['mIoU'])} |")
                    filas_csv.append([modo, K, v, m, my["mIoU"], my["mAcc"], my["allAcc"], uno["mIoU"]])
        ref = d["referencia_todo_a_la_mayoritaria"]
        md += ["", f"Referencia (todo a `{ref['clase']}`): mIoU {f3(ref['mIoU'])}, exactitud {f3(ref['exactitud'])}.", ""]
    (sal / "tablas/estilo_sonata.md").write_text("\n".join(md), encoding="utf-8")
    with open(sal / "tablas/estilo_sonata.csv", "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(filas_csv)

    # ============================ FP / FN por clase ============================
    md = [f"# Falsos positivos y negativos por clase (K = {a.k_ref}, asignación 1 a 1)", "",
          "FN = puntos de la clase que se predijeron otra cosa (o quedaron en un grupo sin clase). FP = puntos de otra clase que se "
          "predijeron como esta. Se muestran también como **% del tamaño real de la clase** (FN% = FNR; FP% = sobre-asignación).", ""]
    filas_csv = [["evaluacion", "K", "m3", "modulo", "clase", "reales", "TP", "FP", "FN", "FP_pct_de_reales", "FN_pct_de_reales", "precision", "recall", "F1", "IoU"]]
    for modo, d in datos.items():
        md += [f"## {'Todas las clases' if modo == 'todas' else 'Sin `otro`'}", "", "| M3 | Método | Clase | Reales | TP | FP | FN | FP % | FN % | Precisión | Recall | F1 | IoU |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for v in variantes:
            for m in MODULOS:
                for c, x in get(d, v, a.k_ref, m)["uno_a_uno"]["por_clase"].items():
                    if not x["puntos_reales"]:
                        continue
                    fp, fn, n = x["FP"], x["FN"], x["puntos_reales"]
                    g = lambda q: "-" if q is None else f"{q:.3f}"
                    md.append(f"| {v} | {m} | {c} | {n:,} | {x['TP']:,} | {fp:,} | {fn:,} | {100 * fp / n:.1f} | {100 * fn / n:.1f} | {g(x['precision'])} | {g(x['recall'])} | {g(x['F1'])} | {g(x['IoU'])} |")
                    filas_csv.append([modo, a.k_ref, v, m, c, n, x["TP"], fp, fn, fp / n, fn / n, x["precision"], x["recall"], x["F1"], x["IoU"]])
        md.append("")
    (sal / "tablas/fp_fn_por_clase.md").write_text("\n".join(md), encoding="utf-8")
    with open(sal / "tablas/fp_fn_por_clase.csv", "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(filas_csv)

    # ============================ graficos ============================
    G = sal / "graficos"
    # 1) mIoU por modulo
    fig, ejes = plt.subplots(2, len(Ks), figsize=(5.2 * len(Ks), 8), squeeze=False)
    for i, (modo, d) in enumerate(datos.items()):
        ref = d["referencia_todo_a_la_mayoritaria"]["mIoU"]
        for j, K in enumerate(Ks):
            ax = ejes[i, j]
            for q, v in enumerate(variantes):
                vals = [get(d, v, K, m)["mayoria"]["mIoU"] for m in MODULOS]
                ax.bar(np.arange(len(MODULOS)) + (q - 0.5) * 0.38, vals, 0.38, label=f"M3 {v}", color=COLOR.get(v))
            ax.axhline(ref, color="k", ls="--", lw=1, label="todo a la mayoritaria")
            ax.set_xticks(range(len(MODULOS))); ax.set_xticklabels(MODULOS); ax.set_ylim(0, 0.8)
            ax.set_title(f"{'todas' if modo == 'todas' else 'sin otro'} · K = {K}"); ax.set_ylabel("mIoU (mayoría)")
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    fig.suptitle("mIoU por método: Sonata original vs adaptado (M2 y M4 base no usan f_ssl)"); fig.tight_layout()
    fig.savefig(G / "miou_por_modulo.png", dpi=130); plt.close(fig)

    # 2) IoU por clase (mapa de calor) en K de referencia
    fig, ejes = plt.subplots(1, 2, figsize=(14, 4.8))
    for ax, (modo, d) in zip(ejes, datos.items()):
        clases = d["clases"]
        etiquetas, M = [], []
        for v in variantes:
            for m in MODULOS:
                etiquetas.append(f"{v[:4]}·{m}")
                M.append([get(d, v, a.k_ref, m)["uno_a_uno"]["iou_por_clase"][c] for c in clases])
        im = ax.imshow(np.array(M), vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(clases))); ax.set_xticklabels(clases); ax.set_yticks(range(len(etiquetas))); ax.set_yticklabels(etiquetas)
        for r in range(len(M)):
            for c in range(len(clases)):
                ax.text(c, r, f"{M[r][c]:.2f}", ha="center", va="center", color="w" if M[r][c] < 0.6 else "k", fontsize=8)
        ax.set_title(f"IoU por clase · K = {a.k_ref} · 1 a 1 · {'todas' if modo == 'todas' else 'sin otro'}")
    fig.tight_layout(); fig.savefig(G / "iou_por_clase.png", dpi=130); plt.close(fig)

    # 3) FP y FN por clase (M3 y M4; original vs adaptado)
    d = datos["sin_otro"]; clases = d["clases"]
    fig, ejes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, m in zip(ejes, ("M3", "M4")):
        x = np.arange(len(clases)); w = 0.2
        for q, v in enumerate(variantes):
            pc = get(d, v, a.k_ref, m)["uno_a_uno"]["por_clase"]
            fn = [100 * pc[c]["FN"] / max(pc[c]["puntos_reales"], 1) for c in clases]
            fp = [100 * pc[c]["FP"] / max(pc[c]["puntos_reales"], 1) for c in clases]
            ax.bar(x + (q * 2 - 1.5) * w, fn, w, color=COLOR.get(v), label=f"{v}: FN %")
            ax.bar(x + (q * 2 - 0.5) * w, fp, w, color=COLOR.get(v), alpha=0.45, hatch="//", label=f"{v}: FP %")
        ax.set_xticks(x); ax.set_xticklabels(clases); ax.set_title(f"{m} · K = {a.k_ref} · sin otro"); ax.set_ylabel("% del tamaño real de la clase")
        ax.legend(fontsize=8)
    fig.suptitle("Falsos negativos (liso) y falsos positivos (rayado) por clase"); fig.tight_layout()
    fig.savefig(G / "fp_fn_por_clase.png", dpi=130); plt.close(fig)

    # 4) matrices de confusion (fila = clase real, normalizada)
    fig, ejes = plt.subplots(2, 2, figsize=(12, 10))
    for i, m in enumerate(("M3", "M4")):
        for j, v in enumerate(variantes):
            r = get(d, v, a.k_ref, m)["uno_a_uno"]
            C = np.array(r["confusion_clases"], float); Cn = C / np.maximum(C.sum(1, keepdims=True), 1)
            ax = ejes[i, j]; im = ax.imshow(Cn, vmin=0, vmax=1, cmap="Blues")
            ax.set_xticks(range(len(r["columnas_confusion"]))); ax.set_xticklabels(r["columnas_confusion"], rotation=35, ha="right", fontsize=8)
            ax.set_yticks(range(len(clases))); ax.set_yticklabels(clases, fontsize=8)
            for p in range(Cn.shape[0]):
                for q in range(Cn.shape[1]):
                    if Cn[p, q] >= 0.01:
                        ax.text(q, p, f"{100 * Cn[p, q]:.0f}", ha="center", va="center", fontsize=8, color="w" if Cn[p, q] > 0.5 else "k")
            ax.set_title(f"{m} · M3 {v} · K = {a.k_ref}"); ax.set_xlabel("clase predicha"); ax.set_ylabel("clase real")
    fig.suptitle("Matrices de confusión (% de cada clase real; 1 a 1; sin otro)"); fig.tight_layout()
    fig.savefig(G / "confusion.png", dpi=130); plt.close(fig)
    print("Listo ->", sal)
    for p in sorted(sal.rglob("*")):
        if p.is_file():
            print("  ", p.relative_to(sal))


if __name__ == "__main__":
    main()
