"""Metricas de segmentacion por clase, a partir de una tabla de cruce y una asignacion grupo -> clase.

Las usan `evaluar_m4.py` (grupos de M2/M3/M4), `sonda_lineal.py` (M3, protocolo de Sonata) y
`verificar_etapas.py`. Definiciones (por clase c, sobre puntos):

    TP  puntos de la clase c que se predijeron c
    FP  puntos de OTRA clase que se predijeron c          (falsos positivos)
    FN  puntos de la clase c que se predijeron otra cosa   (falsos negativos; incluye "sin asignar")
    TN  el resto
    IoU        = TP / (TP + FP + FN)
    precision  = TP / (TP + FP)
    recall     = TP / (TP + FN)       (= exactitud de la clase)
    F1         = 2 P R / (P + R)
    FPR        = FP / (FP + TN)       tasa de falsos positivos
    FNR        = FN / (FN + TP)       tasa de falsos negativos = 1 - recall

Globales (las mismas que reporta Sonata en sus tablas de segmentacion semantica):
    mIoU    promedio de IoU sobre las clases presentes
    mAcc    promedio del recall por clase
    allAcc  puntos bien clasificados / total

HyperNCD (Tabla 1 y 2 del paper) reporta el IoU de las clases conocidas, el IoU de las novedosas tras emparejamiento
hungaro y el promedio de todas ("All"). El emparejamiento hungaro se calcula en `evaluar_m4.py` (`uno_a_uno`).
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def _div(a, b):
    return float(a / b) if b > 0 else None


def desde_cruce(T, asignacion, clases):
    """T: (grupos x clases) con puntos por celda. asignacion[k] = clase del grupo k (o -1 = sin asignar)."""
    T = np.asarray(T, np.int64)
    C = len(clases)
    total = int(T.sum())
    real = T.sum(0)
    conf = np.zeros((C, C + 1), np.int64)                 # fila = clase real, columna = clase predicha; ultima = sin asignar
    for k, c in enumerate(asignacion):
        conf[:, c if c >= 0 else C] += T[k]
    por_clase = {}
    for c, nombre in enumerate(clases):
        tp = int(conf[c, c])
        fn = int(real[c] - tp)
        fp = int(conf[:, c].sum() - tp)
        tn = total - tp - fp - fn
        p, r = _div(tp, tp + fp), _div(tp, tp + fn)
        por_clase[nombre] = {
            "puntos_reales": int(real[c]), "TP": tp, "FP": fp, "FN": fn, "TN": int(tn),
            "IoU": _div(tp, tp + fp + fn) if real[c] > 0 else None,
            "precision": p, "recall": r,
            "F1": (2 * p * r / (p + r)) if (p and r) else (0.0 if real[c] > 0 else None),
            "FPR": _div(fp, fp + tn), "FNR": _div(fn, fn + tp),
        }
    pres = [n for n in clases if por_clase[n]["puntos_reales"] > 0]
    iou = [por_clase[n]["IoU"] for n in pres]
    rec = [por_clase[n]["recall"] for n in pres]
    return {
        "mIoU": float(np.mean(iou)) if iou else None,
        "mAcc": float(np.mean(rec)) if rec else None,
        "allAcc": _div(sum(conf[c, c] for c in range(C)), total),
        "por_clase": por_clase,
        "confusion": conf.tolist(),                        # filas: real; columnas: predicha (+ "sin asignar")
        "columnas_confusion": list(clases) + ["sin_asignar"],
    }


def desde_predicciones(pred, verdad, clases):
    """pred y verdad: clase por punto (enteros 0..C-1); pred = -1 = sin asignar. Ignora verdad < 0."""
    ok = verdad >= 0
    C = len(clases)
    T = np.zeros((C + 1, C), np.int64)
    np.add.at(T, (np.where(pred[ok] >= 0, pred[ok], C), verdad[ok]), 1)
    asign = list(range(C)) + [-1]
    return desde_cruce(T, asign, clases)


def quitar_clase(T, clases, nombre):
    """Deja fuera una clase (sus puntos no se evaluan): quita su columna y los grupos que quedan vacios."""
    T = np.asarray(T, np.int64)
    c = clases.index(nombre)
    keep = [i for i in range(len(clases)) if i != c]
    T2 = T[:, keep]
    return T2[T2.sum(1) > 0], [clases[i] for i in keep]


def evaluar_T(T, clases):
    """Todas las metricas de un agrupamiento a partir de su tabla de cruce T (grupos x clases).
    Dos formas de asignar grupos a clases: 'mayoria' (cada grupo a su clase con mas puntos) y 'uno_a_uno' (hungaro)."""
    T = np.asarray(T, np.int64)
    K, C = T.shape
    may = T.argmax(1)
    fila, col = linear_sum_assignment(-T)
    uno = np.full(K, -1)
    uno[fila] = col
    out = {}
    sin_fondo = [i for i, n in enumerate(clases) if n != "otro" and T[:, i].sum() > 0]
    for nombre, asign in (("mayoria", may), ("uno_a_uno", uno)):
        m = desde_cruce(T, asign, clases)
        iou = np.array([m["por_clase"][c]["IoU"] or 0.0 for c in clases])
        out[nombre] = {
            "mIoU": m["mIoU"],
            "mIoU_sin_otro": float(iou[sin_fondo].mean()) if sin_fondo else None,
            "mAcc": m["mAcc"], "allAcc": m["allAcc"], "exactitud": m["allAcc"],
            "iou_por_clase": {c: float(x) for c, x in zip(clases, iou)},
            "por_clase": m["por_clase"],
            "confusion_clases": m["confusion"], "columnas_confusion": m["columnas_confusion"],
            "asignacion": [int(a) for a in asign],
        }
    idx = np.repeat(np.arange(K * C), T.ravel())      # ARI y NMI solo dependen de la tabla de cruce
    g, c = idx // C, idx % C
    out["ARI"] = float(adjusted_rand_score(c, g))
    out["NMI"] = float(normalized_mutual_info_score(c, g))
    out["cruce_grupos_x_clases"] = T.tolist()
    return out
