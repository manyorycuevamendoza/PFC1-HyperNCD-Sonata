"""Compara M2/M3/M4 en la capilla etiquetada con el M2 de SemanticPOSS (corrida anterior) y con el M2 de ArCH.

Lee las dos evaluaciones `sin_otro` (mismo evaluador, mismas etiquetas) y escribe tabla (.md/.csv/.json) y figura.

Uso (desde la raiz del repo):
    python experimentacion/arch_entrenamiento/comparar_m2_m4.py
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
VIEJO = RAIZ / "experimentacion/escena_capilla_etiquetada/resultados/evaluacion/sin_otro/evaluacion.json"
NUEVO = RAIZ / "experimentacion/arch_entrenamiento/resultados/m4_capilla/evaluacion_sin_otro/evaluacion.json"
SAL = RAIZ / "experimentacion/arch_entrenamiento/resultados/m4_capilla"
METODOS = ["M2", "M3", "M4_base", "M4"]
K = "8"

v, n = json.loads(VIEJO.read_text(encoding="utf-8")), json.loads(NUEVO.read_text(encoding="utf-8"))
cols = {  # nombre -> (evaluacion, variante de Sonata)
    "M2 SemanticPOSS + Sonata original": (v, "original"),
    "M2 ArCH + Sonata original": (n, "original"),
    "M2 SemanticPOSS + Sonata adaptado": (v, "adaptado"),
    "M2 ArCH + Sonata adaptado": (n, "adaptado"),
}
filas = []
for nombre, (ev, var) in cols.items():
    for kk in (K, "6", "12"):
        for m in METODOS:
            d = ev["variantes"][var]["K"][kk][m]
            filas.append({"M2": nombre.split(" + ")[0].replace("M2 ", ""), "sonata": var, "K": int(kk), "metodo": m,
                          "mIoU_1a1_pct": round(100 * d["uno_a_uno"]["mIoU"], 1),
                          "mIoU_mayoria_pct": round(100 * d["mayoria"]["mIoU"], 1),
                          "ARI": round(d["ARI"], 3), "NMI": round(d["NMI"], 3)})
(SAL / "comparacion_m2.json").write_text(json.dumps(filas, indent=1, ensure_ascii=False), encoding="utf-8")
with open(SAL / "comparacion_m2.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(filas[0])); w.writeheader(); w.writerows(filas)

def val(m2, var, met, kk, campo):
    return next(r[campo] for r in filas if r["M2"] == m2 and r["sonata"] == var and r["metodo"] == met and r["K"] == kk)

md = ["# M2 de SemanticPOSS contra M2 de ArCH, capilla etiquetada (sin `otro`)", "",
      "mIoU en %, asignación 1 a 1 (y por mayoría entre paréntesis), K = 8. Misma escena, mismo evaluador.", "",
      "| Método | SemanticPOSS + Sonata original | **ArCH** + Sonata original | SemanticPOSS + Sonata adaptado | **ArCH** + Sonata adaptado |", "|---|---|---|---|---|"]
for met in METODOS:
    cel = []
    for m2, var in (("SemanticPOSS", "original"), ("ArCH", "original"), ("SemanticPOSS", "adaptado"), ("ArCH", "adaptado")):
        cel.append(f"{val(m2, var, met, 8, 'mIoU_1a1_pct'):.1f} ({val(m2, var, met, 8, 'mIoU_mayoria_pct'):.1f})")
    md.append(f"| {met} | " + " | ".join(cel) + " |")
(SAL / "comparacion_m2.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md))

fig, ax = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
anch = 0.2
for j, (campo, titulo) in enumerate((("mIoU_1a1_pct", "asignación 1 a 1"), ("mIoU_mayoria_pct", "asignación por mayoría"))):
    for i, (m2, var) in enumerate((("SemanticPOSS", "original"), ("ArCH", "original"), ("SemanticPOSS", "adaptado"), ("ArCH", "adaptado"))):
        ys = [val(m2, var, m, 8, campo) for m in METODOS]
        ax[j].bar(np.arange(4) + (i - 1.5) * anch, ys, anch, label=f"M2 {m2} + Sonata {var}",
                  color=("#bbbbbb", "#2a7fbf", "#dddddd", "#e07b39")[i], edgecolor="k", linewidth=0.5)
    ax[j].set_xticks(range(4)); ax[j].set_xticklabels(METODOS)
    ax[j].set(title=f"mIoU (%) sin «otro», K=8, {titulo}")
ax[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(SAL / "comparacion_m2.png", dpi=130)
