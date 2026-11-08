"""Compara el Modulo 4 (PL13) corrido con el f_ssl ORIGINAL y con el ADAPTADO.

Lee las dos carpetas de resultados (`resultados/m4_original` y `resultados/m4_adaptado`, que
genera `modulo4_real/correr_modulo4_real.py` con el mismo codigo, las mismas regiones y el
mismo f_sem; solo cambia --fssl) y escribe una tabla y una figura.

Todo es no supervisado: dice si el adaptado cambia los grupos y si los vuelve mas o menos
estables. NO dice cual acierta mas (para eso hace falta el recorte etiquetado).

Uso:  python experimentacion/adaptacion_sonata/comparar_m4.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

AQUI = Path(__file__).resolve().parent
RES = AQUI / "resultados"
SALIDA = RES / "m4_comparacion"
SALIDA.mkdir(parents=True, exist_ok=True)

orig = json.loads((RES / "m4_original" / "modulo4_real_PL13.json").read_text(encoding="utf-8"))
adap = json.loads((RES / "m4_adaptado" / "modulo4_real_PL13.json").read_text(encoding="utf-8"))
zo = np.load(RES / "m4_original" / "grupos_PL13.npz")
za = np.load(RES / "m4_adaptado" / "grupos_PL13.npz")
assert np.array_equal(zo["region"], za["region"]), "las regiones deberian ser identicas"
reg = zo["region"]                       # region de cada punto novedoso

filas, salida = [], {}
for K in (4, 8):
    o, a = orig["K"][str(K)], adap["K"][str(K)]
    bloque = {}
    for var in ("M3", "M4"):
        lo, la = zo[f"K{K}_{var}"][reg], za[f"K{K}_{var}"][reg]   # etiqueta por punto
        bloque[var] = {
            "estabilidad_original": o[var]["estabilidad_ari"],
            "estabilidad_adaptado": a[var]["estabilidad_ari"],
            "silueta_fssl_original": o[var]["silueta_fssl"],
            "silueta_fssl_adaptado": a[var]["silueta_fssl"],
            "ari_original_vs_adaptado": float(adjusted_rand_score(lo, la)),
            "nmi_original_vs_adaptado": float(normalized_mutual_info_score(lo, la)),
        }
    bloque["acuerdo_M3_M4"] = {"original": o["acuerdo"]["M3~M4"]["ari"], "adaptado": a["acuerdo"]["M3~M4"]["ari"]}
    bloque["acuerdo_M2_M4"] = {"original": o["acuerdo"]["M2~M4"]["ari"], "adaptado": a["acuerdo"]["M2~M4"]["ari"]}
    bloque["control_sin_cambio"] = {   # M2 y M4_base no usan f_ssl: deben dar lo mismo
        "M2_estab": [o["M2"]["estabilidad_ari"], a["M2"]["estabilidad_ari"]],
        "M4_base_estab": [o["M4_base"]["estabilidad_ari"], a["M4_base"]["estabilidad_ari"]],
    }
    salida[f"K{K}"] = bloque

(SALIDA / "comparacion_m4_PL13.json").write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")

# ---- tabla en markdown ----
md = ["| K | Variante | Estabilidad orig. | Estabilidad adapt. | Silueta f_ssl orig. | Silueta f_ssl adapt. | ARI orig~adapt |",
      "|---|---|---|---|---|---|---|"]
for K in (4, 8):
    for var in ("M3", "M4"):
        b = salida[f"K{K}"][var]
        md.append(f"| {K} | {var} | {b['estabilidad_original']:.3f} | {b['estabilidad_adaptado']:.3f} | "
                  f"{b['silueta_fssl_original']:.3f} | {b['silueta_fssl_adaptado']:.3f} | {b['ari_original_vs_adaptado']:.3f} |")
md += ["", "| K | Acuerdo M3~M4 orig. | adapt. | Acuerdo M2~M4 orig. | adapt. |", "|---|---|---|---|---|"]
for K in (4, 8):
    b = salida[f"K{K}"]
    md.append(f"| {K} | {b['acuerdo_M3_M4']['original']:.3f} | {b['acuerdo_M3_M4']['adaptado']:.3f} | "
              f"{b['acuerdo_M2_M4']['original']:.3f} | {b['acuerdo_M2_M4']['adaptado']:.3f} |")
(SALIDA / "tabla_m4.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))

# ---- figura: estabilidad y mapa de grupos de M4 (K=4) ----
fig, ejes = plt.subplots(1, 3, figsize=(18, 6))
ax = ejes[0]
etiquetas = ["M3 K4", "M4 K4", "M3 K8", "M4 K8"]
vo = [salida["K4"]["M3"]["estabilidad_original"], salida["K4"]["M4"]["estabilidad_original"],
      salida["K8"]["M3"]["estabilidad_original"], salida["K8"]["M4"]["estabilidad_original"]]
va = [salida["K4"]["M3"]["estabilidad_adaptado"], salida["K4"]["M4"]["estabilidad_adaptado"],
      salida["K8"]["M3"]["estabilidad_adaptado"], salida["K8"]["M4"]["estabilidad_adaptado"]]
x = np.arange(4)
ax.bar(x - 0.2, vo, 0.4, label="f_ssl original", color="#7a8ca5")
ax.bar(x + 0.2, va, 0.4, label="f_ssl adaptado", color="#e07b39")
ax.set_xticks(x); ax.set_xticklabels(etiquetas); ax.set_ylim(0, 1); ax.set_ylabel("Estabilidad (ARI entre semillas)")
ax.set_title("Estabilidad de los grupos"); ax.legend()
cen = zo["cen"]
for ax, z, t in ((ejes[1], zo, "M4 con f_ssl ORIGINAL (K=4)"), (ejes[2], za, "M4 con f_ssl ADAPTADO (K=4)")):
    ax.scatter(cen[:, 0], cen[:, 1], c=z["K4_M4"], cmap="tab10", s=6, linewidths=0)
    ax.set_title(t); ax.set_aspect("equal"); ax.axis("off")
fig.suptitle("Modulo 4 sobre PL13: Sonata original vs adaptado (sin etiquetas: no dice cual acierta mas)")
fig.tight_layout()
fig.savefig(SALIDA / "comparacion_m4_PL13.png", dpi=130)
print("figura ->", SALIDA / "comparacion_m4_PL13.png")
