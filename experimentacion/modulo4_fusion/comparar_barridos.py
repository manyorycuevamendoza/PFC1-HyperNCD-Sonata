"""Junta los barridos de fusion de las 4 combinaciones (M2 x Sonata) y busca configuraciones que ganen de forma CONSISTENTE.

Una fusion es creible si mejora a la base (el M4 actual) en las 4 combinaciones y en las dos mitades de la escena, no solo
en una. Lee `resultados/<nombre>/resultados.json` y escribe tabla (.md/.csv) y figura en `resultados/`.

Uso (desde la raiz del repo):
    python experimentacion/modulo4_fusion/comparar_barridos.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent / "resultados"
NOMBRES = {"semposs_original": "HyperNCD original + Sonata original", "semposs_adaptado": "HyperNCD original + Sonata adaptado",
           "arch_original": "HyperNCD adaptado + Sonata original", "arch_adaptado": "HyperNCD adaptado + Sonata adaptado"}
KP = 8

filas, honestos = [], {}
for n in NOMBRES:
    d = json.loads((AQUI / n / "resultados.json").read_text(encoding="utf-8"))
    honestos[n] = d["honesto"]
    for r in d["resultados"]:
        if r["K"] == KP:
            filas.append({"combo": n, "id": r["id"], "pre": r["pre"], "modo": r["modo"], "a": r["a"], "a_geo": r["a_geo"],
                          "mIoU": 100 * r["mIoU_full"], "A": 100 * r["mIoU_A"], "B": 100 * r["mIoU_B"], "std": 100 * r["std_full"],
                          "estab": r["estabilidad_ari"], **{f"IoU_{c}": 100 * v for c, v in r["iou_por_clase"].items()}})
D = pd.DataFrame(filas)
D = D[D.a_geo == 0.5]            # solo las 39 configuraciones comunes (el 2.o paso, con otro peso de geometria, no existe en todas)
base = D[D.id == "base_M4_actual"].set_index("combo")
D["delta"] = D.apply(lambda r: r.mIoU - base.loc[r.combo, "mIoU"], axis=1)
D["dA"] = D.apply(lambda r: r.A - base.loc[r.combo, "A"], axis=1)
D["dB"] = D.apply(lambda r: r.B - base.loc[r.combo, "B"], axis=1)
D.to_csv(AQUI / "comparacion_combos.csv", index=False)

P = D[D.id != "base_M4_actual"].pivot_table(index="id", columns="combo", values="delta")
P = P[[n for n in NOMBRES if n in P.columns]]
P["media"] = P.mean(1)
P["peor"] = P[[n for n in NOMBRES]].min(1)
P["gana_en"] = (P[[n for n in NOMBRES]] > 0).sum(1)
mitades = D[D.id != "base_M4_actual"].groupby("id")[["dA", "dB"]].apply(lambda g: int(((g.dA > 0) & (g.dB > 0)).sum()))
P["mitades_ok"] = mitades                                  # de 8 (4 combos x 2 mitades)... cuenta combos con A y B arriba
P = P.sort_values(["peor", "media"], ascending=False)

ref = D[D.id == "base_M4_actual"].set_index("combo")
md = ["# Fusiones de f_sem y f_ssl: ¿alguna gana de forma consistente?", "",
      f"mIoU 1 a 1, K = {KP}, sin `otro`, media de 5 semillas. Cada celda = **cambio en puntos respecto a la base** (el M4 actual) de esa misma combinación.", "",
      "Base (M4 actual): " + " · ".join(f"{NOMBRES[n]} **{ref.loc[n, 'mIoU']:.1f}**" for n in NOMBRES), "",
      "| Configuración | " + " | ".join(NOMBRES[n] for n in NOMBRES) + " | media | peor caso | gana en (de 4) | combos con A y B arriba |",
      "|---|" + "---|" * (len(NOMBRES) + 4)]
for i, r in P.head(15).iterrows():
    md.append(f"| `{i}` | " + " | ".join(f"{r[n]:+.1f}" for n in NOMBRES) + f" | {r.media:+.1f} | {r.peor:+.1f} | {int(r.gana_en)} | {int(r.mitades_ok)} |")
md += ["", "## Protocolo honesto por combinación (elegir en una mitad, reportar en la otra)", "",
       "| Combinación | base (media A,B) | estimación honesta | mejora (puntos) | elegida en A | elegida en B |", "|---|---|---|---|---|---|"]
for n in NOMBRES:
    h = honestos[n]
    md.append(f"| {NOMBRES[n]} | {100 * h['base_media_AB']:.1f} | {100 * h['estimacion_honesta']:.1f} | {h['mejora_honesta_puntos']:+.1f} | `{h['elegida_en_A']['id']}` | `{h['elegida_en_B']['id']}` |")
md += ["", "## Medias por familia (cambio respecto a la base, puntos; media de las 4 combinaciones)", ""]
fam = D[D.id != "base_M4_actual"].groupby(["pre", "modo"]).delta.agg(["mean", "max", "min"]).round(1).sort_values("mean", ascending=False)
md += ["| Preprocesado | Modo | media | mejor | peor |", "|---|---|---|---|---|"] + [f"| {a} | {b} | {r['mean']:+.1f} | {r['max']:+.1f} | {r['min']:+.1f} |" for (a, b), r in fam.iterrows()]
(AQUI / "comparacion_combos.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md))

# figura: delta de las 12 mejores por peor caso, una barra por combinacion
top = P.head(12).index
fig, ax = plt.subplots(figsize=(13, 5))
w = 0.2
for j, n in enumerate(NOMBRES):
    ax.bar(np.arange(len(top)) + (j - 1.5) * w, [P.loc[i, n] for i in top], w, label=NOMBRES[n], color=("#b8c2d1", "#f0b48a", "#2a7fbf", "#e07b39")[j], edgecolor="k", linewidth=0.4)
ax.axhline(0, color="k", lw=1)
ax.set_xticks(range(len(top)))
ax.set_xticklabels(top, rotation=40, ha="right", fontsize=8)
ax.set_ylabel("mIoU respecto a la base (puntos)")
ax.set_title(f"Fusiones de f_sem y f_ssl: cambio frente al M4 actual (K={KP}, 1 a 1, sin «otro»; ordenadas por su peor caso)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(AQUI / "comparacion_combos.png", dpi=130)
