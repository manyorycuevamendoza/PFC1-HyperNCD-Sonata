"""Recalcula TODAS las metricas desde las tablas de cruce (grupos x clases) guardadas en un `evaluacion.json`.

Sirve cuando no se tienen los grupos por punto de M4 (por ejemplo, el cuaderno se detuvo antes de copiarlos a Drive) pero
si el `evaluacion.json` que dejo `evaluar_m4.py`. Toda metrica por clase (TP, FP, FN, IoU, precision, recall, F1, FPR, FNR),
mAcc, allAcc, ARI y NMI se puede calcular solo con esa tabla. Tambien hace la evaluacion SIN la clase `otro`, quitando su
columna (equivale a tratar sus puntos como sin etiqueta).

Escribe dos carpetas dentro de --salida:  todas/  y  sin_otro/  (cada una con evaluacion.json, por_clase.csv, evaluacion.md
y evaluacion_miou.png) y un control: el mIoU recalculado debe coincidir con el guardado.

Uso:
    python experimentacion/evaluacion_etiquetada/reevaluar_desde_cruces.py \
        --evaluacion experimentacion/escena_capilla_etiquetada/datos/m4/salida/capilla_etiquetada/evaluacion/evaluacion.json \
        --salida experimentacion/escena_capilla_etiquetada/resultados/evaluacion
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluar_m4 import escribir_salidas  # noqa: E402
from metricas import evaluar_T, quitar_clase  # noqa: E402


def referencia(clases, puntos):
    total = np.array([puntos[c] for c in clases], float)
    k = int(total.argmax())
    iou = np.zeros(len(clases))
    iou[k] = total[k] / total.sum()
    return {"clase": clases[k], "mIoU": float(iou[total > 0].mean()), "exactitud": float(total[k] / total.sum())}


def construir(viejo, quitar=None):
    clases = viejo["clases"]
    out = {"clases": clases, "variantes": {}}
    maxdif = 0.0
    for nombre, bloque in viejo["variantes"].items():
        nuevo = {"cobertura_por_clase": bloque["cobertura_por_clase"], "K": {}}
        for K, mods in bloque["K"].items():
            for var, m in mods.items():
                T = np.array(m["cruce_grupos_x_clases"], np.int64)
                cl = clases
                if quitar:
                    T, cl = quitar_clase(T, clases, quitar)
                r = evaluar_T(T, cl)
                nuevo["K"].setdefault(int(K), {})[var] = r
                if not quitar:   # control contra lo guardado
                    maxdif = max(maxdif, abs(r["mayoria"]["mIoU"] - m["mayoria"]["mIoU"]), abs(r["ARI"] - m["ARI"]))
                nuevo["puntos_evaluados"] = int(T.sum())
        nuevo["cobertura_total"] = bloque["cobertura_total"]
        out["variantes"][nombre] = nuevo
    cl = [c for c in clases if c != quitar]
    out["clases"] = cl
    out["puntos_por_clase"] = {c: viejo["puntos_por_clase"][c] for c in cl}
    out["referencia_todo_a_la_mayoritaria"] = referencia(cl, out["puntos_por_clase"])
    return out, maxdif


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evaluacion", required=True)
    ap.add_argument("--salida", required=True)
    a = ap.parse_args()
    viejo = json.loads(Path(a.evaluacion).read_text(encoding="utf-8"))
    sal = Path(a.salida)

    todas, dif = construir(viejo)
    print(f"Control: diferencia maxima entre lo recalculado y lo guardado (mIoU y ARI) = {dif:.2e}")
    assert dif < 1e-9, "Lo recalculado no coincide con lo guardado: revisa el evaluacion.json"
    escribir_salidas(todas, sal / "todas", [])

    sin, _ = construir(viejo, quitar="otro")
    escribir_salidas(sin, sal / "sin_otro", ["otro"])
    print("\nListo ->", sal)


if __name__ == "__main__":
    main()
