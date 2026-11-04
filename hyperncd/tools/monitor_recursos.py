"""Lanza un comando y muestrea sus recursos cada `--interval` segundos.

Uso:
    python tools/monitor_recursos.py --out profiling/recursos.csv -- python train.py ...

Columnas del CSV:
  GPU (NVML):   util. SM %, util. memoria %, VRAM usada total y del proceso (MB),
                potencia (W), temperatura (C), reloj SM (MHz)
  Proceso:      CPU % (suma del árbol de procesos, 100 = 1 núcleo), RSS (MB),
                hilos, procesos (principal + workers del DataLoader)
  Sistema:      CPU % medio, RAM usada (MB)
"""
import argparse
import csv
import os
import subprocess
import sys
import time

import psutil
import pynvml


def tree(proc):
    try:
        return [proc] + proc.children(recursive=True)
    except psutil.NoSuchProcess:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if not cmd:
        ap.error("falta el comando a monitorizar (después de --)")

    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(args.gpu)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fields = ["time", "gpu_util", "gpu_mem_util", "vram_total_MB", "vram_proc_MB", "power_W",
              "temp_C", "sm_clock_MHz", "proc_cpu_pct", "proc_rss_MB", "proc_threads",
              "proc_count", "sys_cpu_pct", "sys_ram_MB"]

    child = subprocess.Popen(cmd)
    root = psutil.Process(child.pid)
    cpu_seen = {}  # pid -> Process, para que cpu_percent mida entre muestras
    psutil.cpu_percent(None)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        while child.poll() is None:
            time.sleep(args.interval)
            procs = tree(root)
            pids = {p.pid for p in procs}
            cpu = rss = threads = 0
            for p in procs:
                q = cpu_seen.setdefault(p.pid, p)
                try:
                    cpu += q.cpu_percent(None)
                    rss += q.memory_info().rss
                    threads += q.num_threads()
                except psutil.NoSuchProcess:
                    pass
            vram_proc = 0
            for gp in pynvml.nvmlDeviceGetComputeRunningProcesses(h):
                if gp.pid in pids and gp.usedGpuMemory:
                    vram_proc += gp.usedGpuMemory
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            w.writerow({
                "time": round(time.time(), 2),
                "gpu_util": util.gpu,
                "gpu_mem_util": util.memory,
                "vram_total_MB": round(mem.used / 2**20),
                "vram_proc_MB": round(vram_proc / 2**20),
                "power_W": round(pynvml.nvmlDeviceGetPowerUsage(h) / 1000, 1),
                "temp_C": pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU),
                "sm_clock_MHz": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
                "proc_cpu_pct": round(cpu, 1),
                "proc_rss_MB": round(rss / 2**20),
                "proc_threads": threads,
                "proc_count": len(procs),
                "sys_cpu_pct": psutil.cpu_percent(None),
                "sys_ram_MB": round(psutil.virtual_memory().used / 2**20),
            })
            f.flush()
    pynvml.nvmlShutdown()
    sys.exit(child.returncode)


if __name__ == "__main__":
    main()
