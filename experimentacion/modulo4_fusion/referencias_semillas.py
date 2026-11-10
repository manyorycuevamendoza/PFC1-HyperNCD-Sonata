"""M2, M3, M4 base y M4 con VARIAS semillas (media y desviacion), para las 4 combinaciones (HyperNCD x Sonata).

Las evaluaciones anteriores (`evaluar_m4.py`) usan UNA semilla (la mas central de 3). Con un `f_sem` colapsado M4 es muy
inestable, asi que una semilla puede engañar. Aqui se repite cada variante con N semillas sobre los descriptores por region
ya cacheados por `barrido_fusion.py` (resultados/<combo>/cache_regiones.npz) y se reporta media ± desviacion.

Uso (desde la raiz del repo):
    python experimentacion/modulo4_fusion/referencias_semillas.py --semillas 10
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from barrido_fusion import evaluar, m4_regiones, normalizar, tabla  # noqa: E402

COMBOS = {"semposs_original": "HyperNCD original + Sonata original", "semposs_adaptado": "HyperNCD original + Sonata adaptado",
          "arch_original": "HyperNCD adaptado + Sonata original", "arch_adaptado": "HyperNCD adaptado + Sonata adaptado"}

ap = argparse.ArgumentParser()
ap.add_argument("--semillas", type=int, default=10)
ap.add_argument("--K", type=int, nargs="+", default=[8])
a = ap.parse_args()

salida = {}
for nombre in COMBOS:
    z = np.load(AQUI / "resultados" / nombre / "cache_regiones.npz")
    Fsem, Fssl, g, tam, cnt = (z[k] for k in ("Fsem", "Fssl", "g", "tam", "cnt"))
    clases = [str(c) for c in z["clases"]]
    quitar = [clases.index("otro")]
    variantes = {"M2": Fsem, "M3": Fssl,
                 "M4_base": np.hstack([normalizar(Fsem), normalizar(g[:, :3])]),
                 "M4": np.hstack([normalizar(Fsem), normalizar(Fssl)])}
    salida[nombre] = {}
    for K in a.K:
        for v, F in variantes.items():
            m, iou = [], []
            for s in range(a.semillas):
                ev = evaluar(tabla(m4_regiones(F, g, tam, K, np.random.default_rng(s)), K, cnt), quitar)["uno_a_uno"]
                m.append(100 * ev[0])
                iou.append(100 * ev[1])
            salida[nombre].setdefault(str(K), {})[v] = {"media": float(np.mean(m)), "std": float(np.std(m)), "min": float(np.min(m)), "max": float(np.max(m)),
                                                       "iou_por_clase": {c: float(np.mean([x[i] for x in iou])) for i, c in enumerate(clases) if i not in quitar}}
            print(f"{nombre:<17} K={K} {v:<8} {np.mean(m):5.1f} ± {np.std(m):4.1f}  (min {np.min(m):.1f}, max {np.max(m):.1f})", flush=True)

(AQUI / "resultados" / "referencias_semillas.json").write_text(json.dumps({"semillas": a.semillas, "datos": salida}, indent=1, ensure_ascii=False), encoding="utf-8")
Kp = str(a.K[0])
md = [f"# M2, M3, M4 base y M4 con {a.semillas} semillas (K = {Kp}, mIoU 1 a 1, sin `otro`)", "",
      "Media ± desviación sobre semillas de inicialización (y [mín, máx]). Misma escena, mismas etiquetas.", "",
      "| Combinación | M2 | M3 | M4 base (`f_geo`) | M4 (`f_sem` + `f_ssl`) |", "|---|---|---|---|---|"]
for n, t in COMBOS.items():
    d = salida[n][Kp]
    md.append(f"| {t} | " + " | ".join(f"{d[v]['media']:.1f} ± {d[v]['std']:.1f} [{d[v]['min']:.0f}, {d[v]['max']:.0f}]" for v in ("M2", "M3", "M4_base", "M4")) + " |")
(AQUI / "resultados" / "referencias_semillas.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md))
