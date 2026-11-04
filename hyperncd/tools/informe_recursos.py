"""Genera el informe de recursos a partir de una carpeta de perfilado.

Uso:
    python tools/informe_recursos.py profiling/

Lee recursos.csv (tools/monitor_recursos.py), steps.csv y modelo.json
(utils/profiling.py) y escribe en la misma carpeta:
    resumen.json, informe.md, recursos.png, pasos.png
"""
import csv
import json
import math
import os
import statistics as st
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pynvml  # noqa: E402


def read_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        out.append({k: (float(v) if v not in ("", "nan") else math.nan) for k, v in r.items()})
    return out


def col(rows, k):
    return [r[k] for r in rows if not math.isnan(r[k])]


def stats(xs):
    if not xs:
        return {}
    xs_sorted = sorted(xs)
    return {"media": round(st.mean(xs), 2), "mediana": round(st.median(xs), 2),
            "p95": round(xs_sorted[int(0.95 * (len(xs) - 1))], 2), "max": round(max(xs), 2)}


def main(d):
    rec = read_csv(os.path.join(d, "recursos.csv"))
    steps = read_csv(os.path.join(d, "steps.csv"))
    with open(os.path.join(d, "modelo.json")) as f:
        modelo = json.load(f)

    # Solo muestras con el entrenamiento ya en GPU (descarta arranque y cierre)
    busy = [r for r in rec if r["vram_proc_MB"] > 1000] or rec
    # Pasos "completos": descarta el primero de cada época (calentamiento) y el último (batch parcial)
    n_full = max(col(steps, "n_puntos"))
    full = [s for s in steps if s["n_puntos"] >= 0.9 * n_full and not math.isnan(s["data_wait_s"])]

    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    cores = pynvml.nvmlDeviceGetNumGpuCores(h)
    vram_total = pynvml.nvmlDeviceGetMemoryInfo(h).total / 2**20
    pynvml.nvmlShutdown()
    clk = st.mean(col(busy, "sm_clock_MHz"))
    peak_tflops = 2 * cores * clk * 1e6 / 1e12

    fl = modelo["pasos_con_flops"]
    gflops_step = st.mean(r["gflops_total"] for r in fl) if fl else math.nan
    batch_s = st.mean(col(full, "batch_s"))
    tflops_s = gflops_step / batch_s / 1e3

    resumen = {
        "hardware": {"gpu": modelo["gpu"], "nucleos_cuda": cores, "vram_total_MB": round(vram_total),
                     "reloj_sm_medio_MHz": round(clk), "pico_fp32_teorico_TFLOPs": round(peak_tflops, 1)},
        "software": {"torch": modelo["torch"], "cuda": modelo["cuda"]},
        "config": {"batch_size": modelo["batch_size"], "num_workers": modelo["num_workers"],
                   "torch_threads": modelo["torch_threads"], "omp_num_threads": modelo["omp_num_threads"],
                   "puntos_por_paso": int(n_full)},
        "modelo": {"parametros": modelo["parametros"], "parametros_entrenables": modelo["parametros_entrenables"],
                   "convs_dispersas": modelo["convs_dispersas"]},
        "flops_por_paso_GFLOP": {
            "sparse_forward": round(st.mean(r["gflops_sparse_fwd"] for r in fl), 1),
            "sparse_fwd_bwd_estimado": round(st.mean(r["gflops_sparse_total_est"] for r in fl), 1),
            "densas_fwd_bwd": round(st.mean(r["gflops_densas"] for r in fl), 1),
            "total": round(gflops_step, 1),
        },
        "rendimiento": {"s_por_paso": stats(col(full, "batch_s")),
                        "espera_datos_s": stats(col(full, "data_wait_s")),
                        "TFLOPs_por_s": round(tflops_s, 2),
                        "uso_del_pico_fp32_pct": round(100 * tflops_s / peak_tflops, 1),
                        "epocas_s": modelo.get("epocas_s"), "fit_total_s": modelo.get("fit_s")},
        "gpu": {"util_sm_pct": stats(col(busy, "gpu_util")),
                "util_memoria_pct": stats(col(busy, "gpu_mem_util")),
                "vram_proceso_MB": stats(col(busy, "vram_proc_MB")),
                "vram_alloc_pico_torch_MB": round(max(col(steps, "vram_alloc_pico_MB"))),
                "vram_reservada_pico_torch_MB": round(max(col(steps, "vram_reservada_pico_MB"))),
                "potencia_W": stats(col(busy, "power_W")), "temp_C": stats(col(busy, "temp_C"))},
        "cpu": {"cpu_proceso_pct_100_1nucleo": stats(col(busy, "proc_cpu_pct")),
                "hilos": stats(col(busy, "proc_threads")),
                "procesos": stats(col(busy, "proc_count")),
                "rss_MB": stats(col(busy, "proc_rss_MB")),
                "cpu_sistema_pct": stats(col(busy, "sys_cpu_pct")),
                "ram_sistema_MB": stats(col(busy, "sys_ram_MB"))},
        "muestras": {"recursos": len(rec), "pasos": len(steps)},
    }
    with open(os.path.join(d, "resumen.json"), "w") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    # Gráficos
    t0 = rec[0]["time"]
    t = [(r["time"] - t0) for r in rec]
    fig, ax = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    ax[0].plot(t, [r["gpu_util"] for r in rec], label="SM %")
    ax[0].plot(t, [r["gpu_mem_util"] for r in rec], label="memoria %")
    ax[0].set_ylabel("Uso GPU (%)"); ax[0].legend(loc="upper right")
    ax[1].plot(t, [r["vram_proc_MB"] / 1024 for r in rec], label="proceso")
    ax[1].axhline(vram_total / 1024, ls="--", c="gray", label="total GPU")
    ax[1].set_ylabel("VRAM (GB)"); ax[1].legend(loc="upper right")
    ax[2].plot(t, [r["proc_cpu_pct"] / 100 for r in rec], label="núcleos usados")
    ax2 = ax[2].twinx(); ax2.plot(t, [r["proc_threads"] for r in rec], c="C3", label="hilos")
    ax[2].set_ylabel("CPU (núcleos)"); ax2.set_ylabel("hilos", color="C3")
    ax[3].plot(t, [r["proc_rss_MB"] / 1024 for r in rec], label="RAM proceso")
    ax3 = ax[3].twinx(); ax3.plot(t, [r["power_W"] for r in rec], c="C2")
    ax[3].set_ylabel("RAM (GB)"); ax3.set_ylabel("potencia (W)", color="C2")
    ax[3].set_xlabel("tiempo (s)")
    fig.tight_layout(); fig.savefig(os.path.join(d, "recursos.png"), dpi=120); plt.close(fig)

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    sx = [s["step"] for s in steps]
    ax[0].bar(sx, [s["batch_s"] for s in steps]); ax[0].set_ylabel("s / paso")
    ax[1].plot(sx, [s["vram_alloc_pico_MB"] / 1024 for s in steps], label="asignada (pico)")
    ax[1].plot(sx, [s["vram_reservada_pico_MB"] / 1024 for s in steps], label="reservada (pico)")
    ax[1].set_ylabel("VRAM PyTorch (GB)"); ax[1].set_xlabel("paso"); ax[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(d, "pasos.png"), dpi=120); plt.close(fig)

    r = resumen
    md = f"""# Informe de recursos — entrenamiento HyperNCD (M2)

**Hardware:** {r['hardware']['gpu']} ({cores} núcleos CUDA, {vram_total/1024:.1f} GB), pico FP32 teórico
≈ {peak_tflops:.1f} TFLOP/s al reloj medio observado ({clk:.0f} MHz). PyTorch {r['software']['torch']}, CUDA {r['software']['cuda']}.
**Config:** batch {r['config']['batch_size']}, {r['config']['num_workers']} workers, {int(n_full):,} puntos/paso (2 vistas).

| Métrica | Valor |
|---|---|
| Parámetros | {r['modelo']['parametros']:,} |
| FLOPs por paso (fwd+bwd) | {gflops_step/1e3:.2f} TFLOP |
| — convs dispersas (forward, exacto) | {r['flops_por_paso_GFLOP']['sparse_forward']:.0f} GFLOP |
| — operaciones densas (fwd+bwd) | {r['flops_por_paso_GFLOP']['densas_fwd_bwd']:.0f} GFLOP |
| Tiempo por paso (mediana) | {r['rendimiento']['s_por_paso']['mediana']} s |
| Espera de datos (mediana) | {r['rendimiento']['espera_datos_s']['mediana']} s |
| Rendimiento | {tflops_s:.2f} TFLOP/s ({r['rendimiento']['uso_del_pico_fp32_pct']}% del pico) |
| Uso SM GPU (media / p95) | {r['gpu']['util_sm_pct']['media']} % / {r['gpu']['util_sm_pct']['p95']} % |
| VRAM proceso (máx) | {r['gpu']['vram_proceso_MB']['max']/1024:.1f} GB |
| VRAM PyTorch pico asignada / reservada | {r['gpu']['vram_alloc_pico_torch_MB']/1024:.1f} / {r['gpu']['vram_reservada_pico_torch_MB']/1024:.1f} GB |
| Potencia (media / máx) | {r['gpu']['potencia_W']['media']} / {r['gpu']['potencia_W']['max']} W |
| CPU del proceso (media / máx) | {r['cpu']['cpu_proceso_pct_100_1nucleo']['media']/100:.1f} / {r['cpu']['cpu_proceso_pct_100_1nucleo']['max']/100:.1f} núcleos |
| Hilos (media / máx) | {r['cpu']['hilos']['media']:.0f} / {r['cpu']['hilos']['max']:.0f} |
| RAM del proceso, suma RSS (máx; sobreestima la memoria compartida con los workers) | {r['cpu']['rss_MB']['max']/1024:.1f} GB |
| RAM usada del sistema (máx) | {r['cpu']['ram_sistema_MB']['max']/1024:.1f} GB |

![recursos](recursos.png)

![pasos](pasos.png)

Los FLOPs de las convoluciones dispersas cuentan solo los pares activos del mapa de kernel.
Su backward se estima como 2 veces el forward. Las demás operaciones las mide `FlopCounterMode`.
"""
    with open(os.path.join(d, "informe.md"), "w") as f:
        f.write(md)
    print(json.dumps(resumen, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "profiling")
