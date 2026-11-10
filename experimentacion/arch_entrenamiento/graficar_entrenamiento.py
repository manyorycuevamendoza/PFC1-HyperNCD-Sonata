"""Curvas de la corrida de HyperNCD en ArCH (metrics.csv de Lightning) y comparacion M2 SemanticPOSS contra M2 ArCH.

Uso (desde la raiz del repo):
    python experimentacion/arch_entrenamiento/graficar_entrenamiento.py
Lee  experimentacion/arch_entrenamiento/resultados_colab/   (zip bajado de Colab)
Escribe experimentacion/arch_entrenamiento/resultados/entrenamiento/{curvas.png,resumen.json}
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

AQUI = Path(__file__).resolve().parent
COL = AQUI / "resultados_colab"
SAL = AQUI / "resultados" / "entrenamiento"
SAL.mkdir(parents=True, exist_ok=True)

csvs = sorted((COL / "logs").glob("**/metrics.csv"), key=lambda p: p.stat().st_size)
d = pd.read_csv(csvs[-1])                               # la corrida completa es la mas grande
val = d[d["valid/mIoU"].notna()].drop_duplicates("epoch", keep="last").set_index("epoch")
tr = d[d["train/loss_epoch"].notna()].drop_duplicates("epoch", keep="last").set_index("epoch")
clases = [c for c in d.columns if c.startswith("valid/IoU/")]
nuevas = {"arch", "column", "vault"}

fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
ax[0].plot(tr.index, tr["train/loss_epoch"], marker="o", ms=3)
ax[0].set(title="Pérdida de entrenamiento", xlabel="época", ylabel="loss")
ax[1].plot(val.index, val["valid/mIoU"], marker="o", ms=3, color="C3")
mejor = val["valid/mIoU"].idxmax()
ax[1].axvline(mejor, ls=":", color="gray")
ax[1].set(title=f"mIoU de validación (Test de ArCH), mejor época {mejor}: {val['valid/mIoU'].max():.3f}",
          xlabel="época", ylabel="mIoU")
for c in clases:
    n = c.split("/")[-1]
    ax[2].plot(val.index, val[c], label=n + (" (nueva)" if n in nuevas else ""), ls="--" if n in nuevas else "-")
ax[2].set(title="IoU por clase en validación", xlabel="época")
ax[2].legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(SAL / "curvas.png", dpi=130)

ult = val.iloc[-1]
res = {"epocas": int(val.index.max()) + 1, "mejor_epoca": int(mejor), "mIoU_mejor": float(val["valid/mIoU"].max()),
       "mIoU_ultima": float(ult["valid/mIoU"]),
       "IoU_por_clase_mejor": {c.split("/")[-1]: float(val.loc[mejor, c]) for c in clases},
       "IoU_por_clase_ultima": {c.split("/")[-1]: float(ult[c]) for c in clases},
       "loss_train_primera": float(tr["train/loss_epoch"].iloc[0]), "loss_train_ultima": float(tr["train/loss_epoch"].iloc[-1]),
       "medicion": json.loads((COL / "medicion/corrida_completa/resumen.json").read_text(encoding="utf-8"))}
(SAL / "resumen.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps({k: v for k, v in res.items() if k != "medicion"}, indent=1, ensure_ascii=False))
