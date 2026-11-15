"""Imprime en LaTeX las filas por clase (TP, FP, FN, precision, recall, F1) de M3 y M4 con Sonata original.

Lee el por_clase.csv de la evaluacion sin `otro`, con K = 8 y asignacion uno a uno (una semilla).
Uso: python experimentacion/escena_capilla_etiquetada/tabla_por_clase.py <por_clase.csv>
"""
import csv
import sys

R = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
NOMBRES = {"arco": "Arco", "columna": "Columna", "cupula": "Cúpula", "pared": "Pared", "techo": "Techo"}


def pct(x):
    return "---" if x == "" else f"{float(x) * 100:.1f}".replace(".", "{,}")


def miles(x):
    return f"{int(float(x)):,}".replace(",", "\\,")


for mod, titulo in (("M3", "M3, Sonata original"), ("M4_base", "M4 base"), ("M4", "M4, Sonata original")):
    print(f"\\multicolumn{{7}}{{@{{}}l}}{{\\textit{{{titulo}}}}} \\\\")
    for r in R:
        if r["K"] == "8" and r["asignacion"] == "uno_a_uno" and r["modulo"] == mod and r["m3"] == "original" and r["clase"] in NOMBRES:
            print(f"{NOMBRES.get(r['clase'], r['clase'])} & {miles(r['TP'])} & {miles(r['FP'])} & {miles(r['FN'])} & "
                  f"{pct(r['precision'])} & {pct(r['recall'])} & {pct(r['F1'])} \\\\")
