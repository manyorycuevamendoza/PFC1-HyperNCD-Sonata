"""Perfilado del entrenamiento: tiempos por paso, VRAM pico y FLOPs.

Se activa con `train.py --profile`. Escribe `<profile_dir>/steps.csv` (una fila por
paso de entrenamiento) y `<profile_dir>/modelo.json` (parámetros y FLOPs medidos).

FLOPs:
  * Convoluciones dispersas de MinkowskiEngine: se cuentan exactamente con hooks,
    2 * (pares del mapa de kernel) * C_in * C_out por capa en el forward. El backward
    se estima como 2x el forward (gradiente de entrada + gradiente de pesos).
  * Resto de operaciones densas (cabezas, matmuls, Sinkhorn, convs 1x1 de ME, ...):
    `torch.utils.flop_counter.FlopCounterMode`, que ya incluye su backward.
  Solo se miden en `profile_flop_steps` pasos porque los hooks añaden coste.
"""
import csv
import json
import os
import time

import MinkowskiEngine as ME
import torch
from pytorch_lightning import Callback
from torch.utils.flop_counter import FlopCounterMode


class _SparseConvFlops:
    """Hooks de forward en las convoluciones de ME que suman sus FLOPs."""

    def __init__(self, model):
        self.flops = 0
        self.enabled = False
        self.handles = []
        for m in model.modules():
            if isinstance(m, (ME.MinkowskiConvolution, ME.MinkowskiConvolutionTranspose)):
                self.handles.append(m.register_forward_hook(self._hook))

    def _hook(self, module, inputs, output):
        # Las convs 1x1 usan un mm denso y ya las cuenta FlopCounterMode
        if not self.enabled or module.use_mm:
            return
        inp = inputs[0]
        kg = module.kernel_generator
        kmap = inp.coordinate_manager.kernel_map(
            inp.coordinate_map_key,
            output.coordinate_map_key,
            stride=kg.kernel_stride,
            kernel_size=kg.kernel_size,
            dilation=kg.kernel_dilation,
            region_type=kg.region_type,
            is_transpose=module.is_transpose,
        )
        pairs = sum(v.shape[1] for v in kmap.values())
        self.flops += 2 * pairs * module.in_channels * module.out_channels


class ResourceProfilerCallback(Callback):
    def __init__(self, out_dir, flop_steps=(3, 4, 5)):
        self.out_dir = out_dir
        self.flop_steps = set(flop_steps)
        os.makedirs(out_dir, exist_ok=True)
        self.csv_path = os.path.join(out_dir, "steps.csv")
        self.fields = [
            "time", "epoch", "step", "batch_s", "data_wait_s", "n_puntos",
            "vram_alloc_pico_MB", "vram_reservada_pico_MB",
            "gflops_sparse_fwd", "gflops_sparse_total_est", "gflops_densas", "gflops_total",
        ]
        self._file = None
        self._last_end = None
        self._summary = {"pasos_con_flops": []}

    # -- utilidades
    def _write(self, row):
        self._writer.writerow({k: row.get(k, "") for k in self.fields})
        self._file.flush()

    def on_fit_start(self, trainer, pl_module):
        self._file = open(self.csv_path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fields)
        self._writer.writeheader()
        self._sparse = _SparseConvFlops(pl_module.model)
        n_params = sum(p.numel() for p in pl_module.parameters())
        n_train = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
        self._summary.update({
            "parametros": n_params,
            "parametros_entrenables": n_train,
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "batch_size": pl_module.hparams.batch_size,
            "num_workers": pl_module.hparams.num_workers,
            "torch_threads": torch.get_num_threads(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "convs_dispersas": len(self._sparse.handles),
        })
        self._fit_t0 = time.time()

    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_t0 = time.time()
        self._last_end = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        torch.cuda.synchronize()
        now = time.time()
        self._t0 = now
        self._data_wait = (now - self._last_end) if self._last_end is not None else float("nan")
        torch.cuda.reset_peak_memory_stats()
        self._n_points = int(batch[0].shape[0]) + int(batch[7].shape[0])
        step = trainer.global_step
        self._flop_mode = None
        if step in self.flop_steps:
            self._sparse.flops = 0
            self._sparse.enabled = True
            self._flop_mode = FlopCounterMode(display=False)
            self._flop_mode.__enter__()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        row = {}
        if self._flop_mode is not None:
            self._flop_mode.__exit__(None, None, None)
            self._sparse.enabled = False
            sp_fwd = self._sparse.flops / 1e9
            dense = self._flop_mode.get_total_flops() / 1e9
            row.update({
                "gflops_sparse_fwd": round(sp_fwd, 3),
                "gflops_sparse_total_est": round(3 * sp_fwd, 3),
                "gflops_densas": round(dense, 3),
                "gflops_total": round(3 * sp_fwd + dense, 3),
            })
        torch.cuda.synchronize()
        now = time.time()
        row.update({
            "time": now,
            "epoch": trainer.current_epoch,
            "step": trainer.global_step,
            "batch_s": round(now - self._t0, 4),
            "data_wait_s": round(self._data_wait, 4),
            "n_puntos": self._n_points,
            "vram_alloc_pico_MB": round(torch.cuda.max_memory_allocated() / 2**20, 1),
            "vram_reservada_pico_MB": round(torch.cuda.max_memory_reserved() / 2**20, 1),
        })
        if self._flop_mode is not None:
            row["gflops_por_s"] = row["gflops_total"] / row["batch_s"]
            self._summary["pasos_con_flops"].append(row)
        self._write(row)
        self._last_end = now

    def on_train_epoch_end(self, trainer, pl_module):
        self._summary.setdefault("epocas_s", []).append(round(time.time() - self._epoch_t0, 2))

    def on_fit_end(self, trainer, pl_module):
        self._summary["fit_s"] = round(time.time() - self._fit_t0, 2)
        with open(os.path.join(self.out_dir, "modelo.json"), "w") as f:
            json.dump(self._summary, f, indent=2)
        self._file.close()
