"""Corre un comando y mide tiempo, RAM (proceso y sistema), GPU (memoria y uso) mientras dura.

Uso:
    python monitor_recursos.py --salida DIR [--cada 5] [--nombre etiqueta] -- <comando y argumentos>
Ejemplo:
    python monitor_recursos.py --salida m2/salida/monitor/predecir_PL7 -- \
        /content/miniconda3/envs/hyperncd-env/bin/python predecir_hyperncd.py --input x.npy ...

Guarda en DIR: recursos.csv (una fila cada N s), resumen.json (picos, medias, tiempo) y recursos.png.
El monitor corre en el Python de Colab (necesita psutil y matplotlib, ya instalados); el comando
puede usar otro Python (p. ej. hyperncd-env). Mide el proceso lanzado y todos sus hijos."""
import argparse, csv, json, os, subprocess, sys, time
import psutil


def gpu():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.strip()
        n, u, t, ut = [x.strip() for x in o.splitlines()[0].split(",")]
        return n, float(u), float(t), float(ut)
    except Exception:
        return "", float("nan"), float("nan"), float("nan")


def rss_arbol(p):
    try:
        procs = [p] + p.children(recursive=True)
    except psutil.Error:
        return 0.0
    s = 0
    for q in procs:
        try:
            s += q.memory_info().rss
        except psutil.Error:
            pass
    return s / 2**30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", required=True)
    ap.add_argument("--cada", type=float, default=5.0)
    ap.add_argument("--nombre", default="")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        sys.exit("Falta el comando después de --")
    os.makedirs(a.salida, exist_ok=True)
    t0 = time.time()
    proc = subprocess.Popen(cmd)
    ps = psutil.Process(proc.pid)
    gname = gpu()[0]
    filas = []
    while True:
        fin = proc.poll() is not None
        _, gu, gt, gutil = gpu()
        vm = psutil.virtual_memory()
        filas.append([round(time.time() - t0, 1), round(rss_arbol(ps), 3), round(vm.used / 2**30, 3),
                      round(vm.total / 2**30, 3), gu, gt, gutil, psutil.cpu_percent(None)])
        if fin:
            break
        time.sleep(a.cada)
    cab = ["t_s", "ram_proceso_GB", "ram_sistema_usada_GB", "ram_sistema_total_GB",
           "gpu_mem_usada_MB", "gpu_mem_total_MB", "gpu_uso_pct", "cpu_pct"]
    with open(f"{a.salida}/recursos.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(cab); w.writerows(filas)
    col = lambda i: [r[i] for r in filas if r[i] == r[i]]
    mx = lambda i: max(col(i)) if col(i) else None
    mean = lambda i: round(sum(col(i)) / len(col(i)), 1) if col(i) else None
    res = {"nombre": a.nombre, "comando": " ".join(cmd), "codigo_salida": proc.returncode,
           "duracion_s": round(time.time() - t0, 1), "duracion_min": round((time.time() - t0) / 60, 1),
           "gpu": gname, "ram_proceso_pico_GB": mx(1), "ram_sistema_pico_GB": mx(2), "ram_sistema_total_GB": filas[0][3],
           "gpu_mem_pico_MB": mx(4), "gpu_mem_total_MB": filas[0][5], "gpu_uso_medio_pct": mean(6),
           "cpu_medio_pct": mean(7), "muestras": len(filas), "cada_s": a.cada}
    json.dump(res, open(f"{a.salida}/resumen.json", "w"), indent=1, ensure_ascii=False)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        t = [r[0] / 60 for r in filas]
        fig, ax = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
        ax[0].plot(t, col(1) if len(col(1)) == len(t) else [r[1] for r in filas], label="RAM del proceso"); ax[0].plot(t, [r[2] for r in filas], label="RAM del sistema", alpha=.6)
        ax[0].set_ylabel("RAM (GB)"); ax[0].legend()
        ax[1].plot(t, [r[4] / 1024 for r in filas], color="C2"); ax[1].set_ylabel("Memoria GPU (GB)")
        ax[2].plot(t, [r[6] for r in filas], color="C3", label="GPU"); ax[2].plot(t, [r[7] for r in filas], color="C4", alpha=.6, label="CPU")
        ax[2].set_ylabel("Uso (%)"); ax[2].set_xlabel("minutos"); ax[2].legend()
        fig.suptitle(f"{a.nombre or 'ejecución'}: {res['duracion_min']} min, {gname}")
        fig.tight_layout(); fig.savefig(f"{a.salida}/recursos.png", dpi=110)
    except Exception as e:
        print("no se pudo graficar:", e)
    print("\n=== RECURSOS ===\n" + json.dumps(res, indent=1, ensure_ascii=False))
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
