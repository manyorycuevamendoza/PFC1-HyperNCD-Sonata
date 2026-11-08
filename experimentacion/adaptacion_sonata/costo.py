"""Grafico de costo computacional de la adaptacion de Sonata (y comparacion con Sonata original).

Lee `datos_costo/costo_medido.csv` (cada fila dice de donde sale el dato) y, si existe,
`datos_costo/train.log` (el de Pointcept, para ver el tiempo y la perdida por epoca).
Lo que no se midio queda marcado como PENDIENTE: no se inventa.

Uso:  python experimentacion/adaptacion_sonata/costo.py
"""
import csv
import re
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AQUI = Path(__file__).resolve().parent
filas = {r["etapa"]: r for r in csv.DictReader(open(AQUI / "datos_costo" / "costo_medido.csv", encoding="utf-8"))}
def val(k):
    v = filas[k]["valor"].strip()
    return float(v) if v else None

fig, ejes = plt.subplots(1, 3, figsize=(19, 6))

# --- A: tiempo acumulado por epoca (checkpoints, o train.log si esta) ---
ax = ejes[0]
log = AQUI / "datos_costo" / "train.log"
if log.exists():
    t, perd = [], []
    for l in log.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = re.match(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ .*Train result: loss: ([\d.]+)", l)
        if m:
            t.append(datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")); perd.append(float(m.group(2)))
    minutos = [(x - t[0]).total_seconds() / 60 for x in t]
    ax.plot(range(1, len(minutos) + 1), minutos, marker="o", ms=3, color="#e07b39")
    fuente = "train.log"
else:
    ep = [5, 10, 15, 20, 25, 30]
    hora = ["05:51", "06:08", "06:26", "06:43", "07:01", "07:18"]
    base = datetime.strptime("05:35", "%H:%M")
    minutos = [(datetime.strptime(h, "%H:%M") - base).total_seconds() / 60 for h in hora]
    ax.plot(ep, minutos, marker="o", color="#e07b39")
    fuente = "hora de los checkpoints (inicio estimado 05:35)"
ax.set_xlabel("Epoca"); ax.set_ylabel("Minutos de entrenamiento acumulados")
ax.set_title(f"A. Entrenamiento de la adaptacion en Colab L4\n(~{minutos[-1]:.0f} min en total; fuente: {fuente})", fontsize=10)
ax.grid(alpha=.3)

# --- B: horas de GPU gastadas ---
ax = ejes[1]
etiq = ["Corrida invalida\n(error de loop)", "Corrida valida\n(30 epocas)"]
horas = [val("adaptacion_entrenamiento_invalido") / 60, val("adaptacion_entrenamiento_valido") / 60]
barras = ax.bar(etiq, horas, color=["#b0b0b0", "#e07b39"])
for b, h in zip(barras, horas):
    ax.text(b.get_x() + b.get_width() / 2, h + 0.03, f"{h:.2f} h", ha="center")
ax.set_ylabel("Horas de GPU (Colab L4)"); ax.set_ylim(0, max(horas) * 1.25)
ax.set_title("B. Tiempo de GPU gastado en la adaptacion\n(la corrida invalida es costo de depuracion)", fontsize=10)

# --- C: tabla comparativa ---
ax = ejes[2]; ax.axis("off")
def fmt(k, u):
    v = val(k)
    return "PENDIENTE" if v is None else f"{v:g} {u}"
tabla = [
    ["", "Sonata original", "Sonata adaptado"],
    ["Entrenamiento extra", "0 (preentrenado por Meta)", f"{val('adaptacion_entrenamiento_valido'):g} min en L4"],
    ["Iteraciones", "-", f"{val('adaptacion_entrenamiento_valido_iteraciones'):g}"],
    ["Checkpoints en disco", "-", f"{val('checkpoints_en_disco'):g} GB"],
    ["Extraer f_ssl de PL13", fmt("extraccion_f_ssl_original", "s") + " (A6000)", fmt("extraccion_f_ssl_adaptado", "s") + " (L4)"],
    ["VRAM pico al extraer", fmt("extraccion_f_ssl_original_vram_pico", "GB") + " (A6000)", fmt("extraccion_f_ssl_adaptado_vram_pico", "GB") + " (L4, SDPA)"],
    ["Modulo 4 (CPU laptop)", fmt("modulo4_original", "s"), fmt("modulo4_adaptado", "s")],
]
t = ax.table(cellText=tabla[1:], colLabels=tabla[0], loc="center", cellLoc="left")
t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 2.2)
ax.set_title("C. Costo por etapa (PENDIENTE = aun no medido)", fontsize=10)

fig.suptitle("Costo computacional de adaptar Sonata a la Catedral", fontsize=13)
fig.tight_layout()
salida = AQUI / "resultados" / "costo" / "costo_computacional.png"
fig.savefig(salida, dpi=130)
print("figura ->", salida)
