"""Imprime en LaTeX las filas de las pruebas P1, P2 y P3 a partir de los .csv de resultados.

Uso: python experimentacion/pruebas_parametros/tabla_latex.py experimentacion/pruebas_parametros/resultados
"""
import csv
import sys
from pathlib import Path

base = Path(sys.argv[1])


def coma(x, d=1):
    return f"{float(x):.{d}f}".replace(".", "{,}")


def miles(x):
    return f"{int(float(x)):,}".replace(",", "\\,")


for archivo, titulo in (("P1_filtrado", "P1, filtrado"), ("P2_voxel", "P2, vóxel"), ("P3_normales", "P3, vecinos de las normales")):
    print(f"\\multicolumn{{8}}{{@{{}}l}}{{\\textit{{{titulo}}}}} \\\\")
    for r in csv.DictReader(open(base / f"{archivo}.csv", encoding="utf-8")):
        cond = r["condicion"].replace("lambda=", "$\\lambda=$").replace("q=", "$q=$").replace("(referencia)", "(ref.)")
        print(f"{cond} & {miles(r['n_voxel'])} & {coma(r['pct_descartado'], 2)} & {coma(r['r_mm_global'])} & "
              f"{coma(r['disp_grados_global'])} & {coma(r['disp_grados_ralas'])} & {coma(r['disp_grados_planas'])} & "
              f"{coma(r['disp_grados_aristas'])} \\\\")
