"""Módulo 2 del pipeline: inferencia y extracción de f_sem con HyperNCD sobre la Catedral.

Carga un checkpoint entrenado de HyperNCD (.ckpt de PyTorch Lightning), lee una
nube preprocesada del Módulo 1 (.npy de 9 columnas: XYZ RGB NxNyNz), cuantiza
las coordenadas con MinkowskiEngine y ejecuta MinkUnet en modo evaluación.

Entrega por cada sección (p. ej. PL7 y PL13):
    1. f_sem: descriptor semántico de 96 dimensiones por vóxel (y mapeo inverse
       a los N puntos del .npy, compatible con visor_3d.py y con el Módulo 4).
    2. logits_lab y logits_unlab: predicciones de cabezas conocidas y novedosas.
    3. pred_label: etiqueta predicha por punto y resumen en .json (VRAM, tiempo,
       rendimiento y conteo de puntos por clase).

Uso:
    python predecir_hyperncd.py \
        --input data/PL13_SCN0001.npy \
        --ckpt  hyperncd/checkpoints/SemanticPOSS/split0_run4/SemanticPOSS_S0_best-epoch=8-step=11196.ckpt \
        --output cache/PL13_fsem.npz
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Agregar hyperncd/ al path para importar models.multiheadminkunet
RAIZ_REPO = Path(__file__).resolve().parent
RUTA_HYPERNCD = RAIZ_REPO / "hyperncd"
if str(RUTA_HYPERNCD) not in sys.path:
    sys.path.insert(0, str(RUTA_HYPERNCD))

import MinkowskiEngine as ME
from models.multiheadminkunet import MinkUnet


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", required=True, help="nube del Módulo 1 (.npy o .txt de 9 columnas)")
    ap.add_argument("--ckpt", required=True, help="ruta al checkpoint entrenado (.ckpt) de HyperNCD")
    ap.add_argument("--output", required=True, help="ruta de salida .npz (también guarda resumen .json)")
    ap.add_argument("--voxel", type=float, default=0.05,
                    help="tamaño de vóxel para cuantización MinkowskiEngine en metros (default: 0.05)")
    ap.add_argument("--fp32", action="store_true",
                    help="guardar descriptores f_sem en float32 en vez de float16")
    return ap.parse_args()


def cargar_nube(ruta):
    """Lee el .npy (o .txt) de 9 columnas del Módulo 1 y devuelve la matriz S."""
    ruta = Path(ruta)
    if ruta.suffix == ".npy":
        datos = np.load(ruta).astype(np.float32)
    else:
        datos = np.loadtxt(ruta, dtype=np.float32)
    if datos.ndim != 2 or datos.shape[1] != 9:
        raise ValueError(
            f"{ruta}: se esperaban 9 columnas (XYZ RGB NxNyNz), "
            f"se encontraron {datos.shape}"
        )
    return datos


def cargar_modelo_hyperncd(ruta_ckpt):
    """Carga MinkUnet desde un checkpoint .ckpt de Discoverer (PyTorch Lightning)."""
    ckpt = torch.load(ruta_ckpt, map_location="cpu")
    hparams = ckpt.get("hyper_parameters", {})
    num_labeled = int(hparams.get("num_labeled_classes", 9))
    num_unlabeled = int(hparams.get("num_unlabeled_classes", 4))
    split = int(hparams.get("split", 0))
    epoch = ckpt.get("epoch", -1)
    global_step = ckpt.get("global_step", -1)

    modelo = MinkUnet(
        num_labeled=num_labeled,
        num_unlabeled=num_unlabeled,
        discover=True,
    )

    state_dict = ckpt["state_dict"]
    pesos_modelo = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            pesos_modelo[k[len("model."):]] = v
        else:
            pesos_modelo[k] = v

    faltantes, inesperadas = modelo.load_state_dict(pesos_modelo, strict=False)
    if faltantes:
        print(f"      [aviso] claves faltantes al cargar state_dict: {faltantes}")
    if inesperadas:
        print(f"      [aviso] claves no usadas del ckpt: {inesperadas}")

    modelo = modelo.cuda()
    modelo.eval()
    info = {
        "num_labeled": num_labeled,
        "num_unlabeled": num_unlabeled,
        "split": split,
        "epoch": int(epoch),
        "global_step": int(global_step),
    }
    return modelo, info


def cuantizar_coordenadas(coord, voxel_size):
    """Cuantiza las coordenadas con MinkowskiEngine y devuelve (discrete_coords, unique_map, inverse)."""
    res = ME.utils.sparse_quantize(
        coordinates=np.ascontiguousarray(coord, dtype=np.float32),
        return_index=True,
        return_inverse=True,
        quantization_size=voxel_size,
    )
    if len(res) == 3:
        discrete_coords, unique_map, inverse = res
    else:
        discrete_coords, unique_map = res
        # Respaldo si la versión de ME no devuelve inverse directamente
        q = np.floor(coord / voxel_size).astype(np.int64)
        _, _, inverse = np.unique(q, axis=0, return_index=True, return_inverse=True)

    if isinstance(discrete_coords, torch.Tensor):
        discrete_coords = discrete_coords.cpu().numpy()
    if isinstance(unique_map, torch.Tensor):
        unique_map = unique_map.cpu().numpy()
    if isinstance(inverse, torch.Tensor):
        inverse = inverse.cpu().numpy()

    return (
        discrete_coords.astype(np.int32),
        unique_map.astype(np.int32),
        inverse.astype(np.int32),
    )


def main():
    args = parse_args()

    print(f"[1/4] leyendo {args.input}")
    t0 = time.time()
    S = cargar_nube(args.input)
    t_lectura = time.time() - t0
    coord = np.ascontiguousarray(S[:, 0:3], dtype=np.float32)
    n_original = coord.shape[0]
    print(f"      {n_original:,} puntos ({t_lectura:.2f}s)")

    print(f"[2/4] cargando checkpoint de HyperNCD: {args.ckpt}")
    modelo, info_ckpt = cargar_modelo_hyperncd(args.ckpt)
    print(
        f"      split={info_ckpt['split']} | epoch={info_ckpt['epoch']} | "
        f"step={info_ckpt['global_step']} | "
        f"clases conocidas={info_ckpt['num_labeled']} | "
        f"clases novedosas={info_ckpt['num_unlabeled']}"
    )

    print(f"[3/4] cuantizando con MinkowskiEngine (voxel = {args.voxel} m)")
    discrete_coords, unique_map, inverse = cuantizar_coordenadas(coord, args.voxel)
    n_rejilla = discrete_coords.shape[0]
    pct_retenido = 100.0 * n_rejilla / n_original
    print(f"      {n_original:,} -> {n_rejilla:,} vóxeles ({pct_retenido:.2f}% retenido)")

    print("[4/4] forward en GPU")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        bcoords = ME.utils.batched_coordinates([discrete_coords]).int().cuda()
        bfeats = torch.ones((n_rejilla, 1), dtype=torch.float32, device="cuda")
        sp_tensor = ME.SparseTensor(features=bfeats, coordinates=bcoords)
        out = modelo(sp_tensor)
        torch.cuda.synchronize()
    t_forward = time.time() - t0
    vram_pico = torch.cuda.max_memory_allocated() / (1024 ** 3)

    dtype_feat = np.float32 if args.fp32 else np.float16
    f_sem = out["feats"].detach().cpu().numpy().astype(dtype_feat)           # (M, 96)
    logits_lab = out["logits_lab"].detach().cpu().numpy().astype(dtype_feat)  # (M, num_labeled)
    logits_unlab = out["logits_unlab"].detach().cpu().numpy().astype(dtype_feat)  # (M, num_unlabeled)

    # Concatenar logits [conocidas ; novedosas] como en validation_step de Discoverer_hyper.py
    logits_all = np.concatenate([logits_lab.astype(np.float32), logits_unlab.astype(np.float32)], axis=1)
    pred_label = logits_all.argmax(axis=1).astype(np.int32)
    pred_lab_only = logits_lab.astype(np.float32).argmax(axis=1).astype(np.int32)
    pred_unlab_only = logits_unlab.astype(np.float32).argmax(axis=1).astype(np.int32)
    es_novedosa = (pred_label >= info_ckpt["num_labeled"]).astype(np.int8)

    # Coordenadas desplazadas al convenio de visor_3d.py (centro XY y mínimo Z)
    mn, mx = coord.min(axis=0), coord.max(axis=0)
    centro_visor = np.array([(mn[0] + mx[0]) / 2.0, (mn[1] + mx[1]) / 2.0, mn[2]], dtype=np.float32)
    coord_rejilla_m1 = coord[unique_map].astype(np.float32)
    coord_visor = (coord_rejilla_m1 - centro_visor).astype(np.float32)

    # Estadísticas de predicción sobre los N puntos originales de S (usando inverse)
    pred_por_punto = pred_label[inverse]
    novedosa_por_punto = es_novedosa[inverse]
    clases_unicas, conteos = np.unique(pred_por_punto, return_counts=True)
    resumen_clases = {int(c): int(cnt) for c, cnt in zip(clases_unicas, conteos)}
    n_conocidas = int((novedosa_por_punto == 0).sum())
    n_novedosas = int((novedosa_por_punto == 1).sum())

    print(f"      forward en {t_forward:.2f}s")
    print(f"\n>>> FEATURES SEMÁNTICAS (f_sem): {f_sem.shape}   dim f_sem = {f_sem.shape[1]}")
    print(f">>> VRAM PICO:    {vram_pico:.2f} GB")
    print(f">>> TIEMPO:       lectura {t_lectura:.2f}s | forward {t_forward:.2f}s")
    print(f">>> RENDIMIENTO:  {n_rejilla / max(t_forward, 1e-6):,.0f} vóxeles/s")
    print(
        f">>> PREDICCIÓN:   conocidas={n_conocidas:,} ({100*n_conocidas/n_original:.1f}%) | "
        f"novedosas={n_novedosas:,} ({100*n_novedosas/n_original:.1f}%)"
    )
    print(f">>> CONTEO POR CLASE (sobre {n_original:,} puntos): {resumen_clases}")

    salida = Path(args.output)
    salida.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        salida,
        feat=f_sem,
        coord=coord_visor,
        coord_m1=coord_rejilla_m1,
        inverse=inverse,
        unique_map=unique_map,
        logits_lab=logits_lab,
        logits_unlab=logits_unlab,
        pred_label=pred_label,
        pred_lab_only=pred_lab_only,
        pred_unlab_only=pred_unlab_only,
        es_novedosa=es_novedosa,
        grid_size=np.float32(args.voxel),
        n_original=np.int64(n_original),
    )

    meta = {
        "input": str(args.input),
        "ckpt": str(args.ckpt),
        "ckpt_info": info_ckpt,
        "voxel_size": float(args.voxel),
        "n_original": int(n_original),
        "n_rejilla_me": int(n_rejilla),
        "pct_retenido": round(pct_retenido, 2),
        "dim_fsem": int(f_sem.shape[1]),
        "dtype": "float32" if args.fp32 else "float16",
        "gpu": torch.cuda.get_device_name(0),
        "capacidad_cuda": "%d.%d" % torch.cuda.get_device_capability(0),
        "vram_pico_gb": round(vram_pico, 3),
        "t_lectura_s": round(t_lectura, 3),
        "t_forward_s": round(t_forward, 3),
        "voxeles_por_s": round(n_rejilla / max(t_forward, 1e-6), 1),
        "puntos_clases_conocidas": n_conocidas,
        "puntos_clases_novedosas": n_novedosas,
        "conteo_por_clase": resumen_clases,
    }
    ruta_json = salida.with_suffix(".json")
    ruta_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # Guardar tabla CSV de conteo por clase para incluir directo en tablas de la tesis
    ruta_csv = salida.with_suffix(".csv")
    lineas_csv = ["clase_id,tipo,puntos,porcentaje"]
    for c, cnt in zip(clases_unicas, conteos):
        tipo = "conocida" if int(c) < info_ckpt["num_labeled"] else "novedosa"
        pct = 100.0 * int(cnt) / n_original
        lineas_csv.append(f"{int(c)},{tipo},{int(cnt)},{pct:.4f}")
    ruta_csv.write_text("\n".join(lineas_csv) + "\n", encoding="utf-8")

    print(f"guardado -> {salida}")
    print(f"resumen  -> {ruta_json}")
    print(f"tabla    -> {ruta_csv}")


if __name__ == "__main__":
    main()
