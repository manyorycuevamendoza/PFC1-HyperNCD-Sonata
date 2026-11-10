"""Verifica, etapa por etapa (M1 -> M4), que cada modulo hizo lo que debia, usando las etiquetas reales de la escena.

Entrada: la carpeta `datos/` de la escena (la que deja `traer_resultados.py`).
Salida (en --salida): etapas.json, etapas.md y las figuras m1_clases.png y m2_cabezas.png.

Que se comprueba en cada etapa
  M1  conteos (crudo -> SOR -> voxel), que la clase se conserve, calidad de la transferencia de etiquetas, normales.
  M2  cabezas de HyperNCD contra las clases reales: cuanto se marca como "novedoso", como se reparten los puntos entre
      las 13 cabezas, y metricas (mIoU, mAcc, allAcc, FP y FN por clase) de las cabezas como clasificador PUNTO A PUNTO,
      sin pasar por regiones. Salud de f_sem (rango efectivo).
  M3  que pesos cargo cada variante, memoria y tiempo (de los logs). Las metricas de M3 estan en la evaluacion de M4
      y, para el protocolo de Sonata (sonda lineal), en `sonda_lineal.py`.
  M4  cobertura y numero de regiones (de los logs).
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metricas import evaluar_T, quitar_clase  # noqa: E402


def leer_log(ruta):
    return Path(ruta).read_text(encoding="utf-8", errors="ignore") if Path(ruta).exists() else ""


def rango_efectivo(f, rng, n=100_000):
    idx = rng.choice(len(f), min(len(f), n), replace=False)
    x = f[idx].astype(np.float32)
    _, s, _ = np.linalg.svd(x - x.mean(0), full_matrices=False)
    v = s ** 2 / (s ** 2).sum()
    p = v[v > 0]
    return float(np.exp(-(p * np.log(p)).sum())), float(v[:3].sum()), float(v[:10].sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datos", required=True, help="carpeta datos/ de la escena")
    ap.add_argument("--salida", required=True)
    ap.add_argument("--escena", default="capilla_etiquetada")
    a = ap.parse_args()
    D, E, sal = Path(a.datos), a.escena, Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    res = {"escena": E}

    # ------------------------------------------------------------------ M1
    m1 = json.loads((D / f"m1/salida/{E}/{E}.json").read_text(encoding="utf-8"))
    info = json.loads((D / f"m1/salida/{E}/{E}_etiquetas.json").read_text(encoding="utf-8"))
    clases = info["clases"]
    lab = np.load(D / f"m1/salida/{E}/{E}_etiquetas.npy")
    S = np.load(D / f"m1/salida/{E}/{E}.npy", mmap_mode="r")
    cruda = np.load(D / f"m1/entrada/{E}/{E}_etiquetas.npz", allow_pickle=False)
    raw = np.bincount(cruda["label"], minlength=len(clases))
    fin = np.array([(lab == i).sum() for i in range(len(clases))])
    nz = np.asarray(S[:, 8])
    nn = np.linalg.norm(np.asarray(S[:, 6:9]), axis=1)
    res["M1"] = {
        "puntos": {"crudo": m1["n_crudo"], "tras_SOR": m1["n_sor"], "tras_voxel": m1["n_voxel"],
                   "SOR_descarto": m1["n_crudo"] - m1["n_sor"], "SOR_descarto_pct": 100 * (1 - m1["n_sor"] / m1["n_crudo"]),
                   "reduccion_total_pct": 100 * (1 - m1["n_voxel"] / m1["n_crudo"])},
        "parametros": {k: m1[k] for k in ("voxel", "q", "lambda")}, "segundos": m1["segundos"],
        "por_clase": {c: {"crudo": int(raw[i]), "tras_M1": int(fin[i]), "conservado_pct": float(100 * fin[i] / raw[i]),
                          "parte_antes_pct": float(100 * raw[i] / raw.sum()), "parte_despues_pct": float(100 * fin[i] / fin.sum())}
                      for i, c in enumerate(clases)},
        "etiquetas": {"sin_etiqueta": info["sin_etiqueta"], "dist_mediana_cm": 100 * info["dist_mediana_m"],
                      "dist_p95_cm": 100 * info["dist_p95_m"], "dist_max_cm": 100 * info["dist_max_m"]},
        "normales": {"norma_media": float(nn.mean()), "norma_min": float(nn.min()), "orientadas_hacia_arriba_pct": float(100 * (nz >= 0).mean())},
        "color_max": float(np.asarray(S[::50, 3:6]).max()),
    }
    ok = (np.abs(nn - 1) < 0.01).mean()
    res["M1"]["normales"]["norma_unitaria_pct"] = float(100 * ok)

    # ------------------------------------------------------------------ M2
    z = np.load(D / f"m2/salida/{E}/{E}_fsem.npz")
    j2 = json.loads((D / f"m2/salida/{E}/{E}_fsem.json").read_text(encoding="utf-8"))
    inv = z["inverse"]
    cabeza = z["pred_label"][inv]                    # cabeza de cada punto (0-8 conocidas, 9-12 novedosas)
    novel = z["es_novedosa"][inv] == 1
    nl, nu = j2["ckpt_info"]["num_labeled"], j2["ckpt_info"]["num_unlabeled"]
    H = nl + nu
    T = np.zeros((H, len(clases)), np.int64)
    np.add.at(T, (cabeza, lab), 1)
    m_todas = evaluar_T(T, clases)
    T2, cl2 = quitar_clase(T, clases, "otro")
    m_sin = evaluar_T(T2, cl2)
    rango, v3, v10 = rango_efectivo(z["feat"], rng)
    uso = np.bincount(cabeza, minlength=H)
    res["M2"] = {
        "checkpoint": j2["ckpt"].split("/")[-2] + "/" + j2["ckpt"].split("/")[-1], "ckpt_info": j2["ckpt_info"],
        "gpu": j2.get("gpu"), "vram_pico_gb": j2["vram_pico_gb"], "forward_s": j2["t_forward_s"],
        "cabezas": {"conocidas": nl, "novedosas": nu},
        "puntos_novedosos_pct": float(100 * novel.mean()), "puntos_conocidos": int((~novel).sum()),
        "uso_por_cabeza": {str(h): int(uso[h]) for h in range(H)},
        "cabeza_dominante": {"cabeza": int(uso.argmax()), "pct": float(100 * uso.max() / uso.sum())},
        "novedoso_por_clase_pct": {c: float(100 * novel[lab == i].mean()) for i, c in enumerate(clases)},
        "f_sem": {"rango_efectivo": rango, "var_pca3": v3, "var_pca10": v10},
        "tabla_cabeza_x_clase": T.tolist(),
        "como_clasificador_punto_a_punto": {"todas": m_todas, "sin_otro": m_sin},
    }

    # ------------------------------------------------------------------ M3 y M4 (de los logs)
    logs = D / f"m4/salida/{E}/logs"
    res["M3"], res["M4"] = {}, {}
    for var in ("original", "adaptado"):
        t = leer_log(logs / f"m3_{var}.log")
        cargo = re.search(r"Loading checkpoint (.*?) \.\.\.", t)
        g = lambda p: (re.search(p, t).group(1) if re.search(p, t) else None)
        res["M3"][var] = {"pesos_cargados": cargo.group(1) if cargo else None,
                          "filas_rejilla": g(r"-> (\d+) puntos"), "vram_pico_gb": g(r"VRAM PICO:\s+([\d.]+)"),
                          "forward_s": g(r"forward ([\d.]+)s"), "puntos_por_s": g(r"RENDIMIENTO:\s+([\d,]+)")}
        t4 = leer_log(logs / f"m4_{var}.log")
        res["M4"][var] = {"puntos_novedosos_pct": (re.search(r"novedosos segun M2 ([\d.]+)%", t4) or [None, None])[1],
                          "regiones": (re.search(r"regiones ([\d,]+)", t4) or [None, None])[1],
                          "cobertura_pct": (re.search(r"cubren ([\d.]+)%", t4) or [None, None])[1],
                          "segundos_total": (re.search(r"\[ok\].*\(([\d.]+)s\)", t4) or [None, None])[1]}
    (sal / "etapas.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")

    # ------------------------------------------------------------------ markdown
    M1, M2 = res["M1"], res["M2"]
    f = lambda x, d=1: "-" if x is None else f"{x:,.{d}f}"
    md = [f"# Verificación por etapa — escena `{E}`", "",
          "Cada bloque dice **qué se comprobó y qué salió**. Los números vienen de las etiquetas reales de la escena.", "",
          "## M1 · Preprocesamiento", "",
          f"- Puntos: crudo **{M1['puntos']['crudo']:,}** → tras SOR **{M1['puntos']['tras_SOR']:,}** "
          f"(descarta {M1['puntos']['SOR_descarto_pct']:.1f} %) → tras voxel de 5 cm **{M1['puntos']['tras_voxel']:,}** "
          f"(reducción total {M1['puntos']['reduccion_total_pct']:.1f} %). Tardó {M1['segundos']} s.",
          f"- Transferencia de etiquetas: mediana {M1['etiquetas']['dist_mediana_cm']:.1f} cm, p95 {M1['etiquetas']['dist_p95_cm']:.1f} cm, "
          f"máximo {M1['etiquetas']['dist_max_cm']:.1f} cm; puntos sin etiqueta: {M1['etiquetas']['sin_etiqueta']}.",
          f"- Normales: norma media {M1['normales']['norma_media']:.3f}; {M1['normales']['norma_unitaria_pct']:.1f} % de norma unitaria; "
          f"{M1['normales']['orientadas_hacia_arriba_pct']:.1f} % orientadas hacia arriba.", "",
          "| Clase | Puntos crudos | Tras M1 | Conservado | Parte antes | Parte después |", "|---|---|---|---|---|---|"]
    for c, d in M1["por_clase"].items():
        md.append(f"| {c} | {d['crudo']:,} | {d['tras_M1']:,} | {d['conservado_pct']:.1f} % | {d['parte_antes_pct']:.1f} % | {d['parte_despues_pct']:.1f} % |")
    md += ["", "Lectura: el voxel reduce más donde hay más densidad; si la **parte** de una clase cambia mucho entre antes y después, M1 la está sesgando.", "",
           "## M2 · HyperNCD (`f_sem`)", "",
           f"- Checkpoint: `{M2['checkpoint']}` (época {M2['ckpt_info']['epoch']}, paso {M2['ckpt_info']['global_step']}).",
           f"- **{M2['puntos_novedosos_pct']:.1f} %** de los puntos se marcan como novedosos; solo {M2['puntos_conocidos']:,} como clase conocida.",
           f"- **Cabeza dominante: la {M2['cabeza_dominante']['cabeza']} con el {M2['cabeza_dominante']['pct']:.1f} % de los puntos** "
           f"(de {nl + nu} cabezas: {nl} conocidas y {nu} novedosas). Reparto: " + ", ".join(f"{h}: {v:,}" for h, v in M2["uso_por_cabeza"].items() if v > 0) + ".",
           f"- Salud de `f_sem`: rango efectivo {M2['f_sem']['rango_efectivo']:.1f}; varianza de 3 y 10 componentes {M2['f_sem']['var_pca3']:.2f} y {M2['f_sem']['var_pca10']:.2f}.",
           f"- VRAM pico {M2['vram_pico_gb']} GB, forward {M2['forward_s']} s" + (f", GPU {M2['gpu']}" if M2.get("gpu") else "") + ".", "",
           "**Las cabezas de M2 como clasificador punto a punto** (sin regiones; asignación por mayoría y 1 a 1):", "",
           "| Evaluación | mIoU (mayoría) | mAcc | allAcc | mIoU 1 a 1 |", "|---|---|---|---|---|"]
    for nombre, m in (("todas las clases", m_todas), ("sin `otro`", m_sin)):
        md.append(f"| {nombre} | {m['mayoria']['mIoU']:.3f} | {m['mayoria']['mAcc']:.3f} | {m['mayoria']['allAcc']:.3f} | {m['uno_a_uno']['mIoU']:.3f} |")
    md += ["", "Porcentaje de cada clase real que M2 marca como novedoso: " + ", ".join(f"{c} {v:.1f} %" for c, v in M2["novedoso_por_clase_pct"].items()) + ".", "",
           "## M3 · Sonata (`f_ssl`)", "", "| Variante | Pesos cargados | Filas de rejilla | VRAM pico (GB) | Forward (s) | Puntos/s |", "|---|---|---|---|---|---|"]
    for var, d in res["M3"].items():
        md.append(f"| {var} | `{d['pesos_cargados']}` | {d['filas_rejilla']} | {d['vram_pico_gb']} | {d['forward_s']} | {d['puntos_por_s']} |")
    md += ["", "Las métricas de M3 con etiquetas están en la evaluación de M4 (módulo `M3`) y en la sonda lineal (`sonda_lineal.py`).", "",
           "## M4 · Fusión e hipergrafo", "", "| M3 usado | Novedosos según M2 | Regiones | Cobertura de los novedosos | Tiempo (s) |", "|---|---|---|---|---|"]
    for var, d in res["M4"].items():
        md.append(f"| {var} | {d['puntos_novedosos_pct']} % | {d['regiones']} | {d['cobertura_pct']} % | {d['segundos_total']} |")
    (sal / "etapas.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------ figuras
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(len(clases))
    ax[0].bar(x - 0.2, raw, 0.4, label="crudo", color="#9aa5b1"); ax[0].bar(x + 0.2, fin, 0.4, label="tras M1", color="#2b7bba")
    ax[0].set_yscale("log"); ax[0].set_xticks(x); ax[0].set_xticklabels(clases); ax[0].set_ylabel("puntos (escala log)"); ax[0].legend()
    ax[0].set_title("M1: puntos por clase antes y después")
    ax[1].bar(x - 0.2, 100 * raw / raw.sum(), 0.4, color="#9aa5b1"); ax[1].bar(x + 0.2, 100 * fin / fin.sum(), 0.4, color="#2b7bba")
    ax[1].set_xticks(x); ax[1].set_xticklabels(clases); ax[1].set_ylabel("% de la escena"); ax[1].set_title("M1: parte de cada clase (¿se sesga?)")
    fig.tight_layout(); fig.savefig(sal / "m1_clases.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1.3, 1]})
    Tn = T / np.maximum(T.sum(1, keepdims=True), 1)
    im = ax[0].imshow(Tn.T, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax[0].set_yticks(range(len(clases))); ax[0].set_yticklabels(clases)
    ax[0].set_xticks(range(H)); ax[0].set_xticklabels([f"{h}{'*' if h >= nl else ''}" for h in range(H)])
    ax[0].set_xlabel("cabeza de M2   (* = novedosa)"); ax[0].set_title("M2: de qué clase real son los puntos de cada cabeza (por columna)")
    fig.colorbar(im, ax=ax[0], fraction=0.03)
    ax[1].bar(range(H), uso, color=["#7a8ca5" if h < nl else "#e07b39" for h in range(H)])
    ax[1].set_yscale("log"); ax[1].set_xlabel("cabeza"); ax[1].set_ylabel("puntos (log)"); ax[1].set_title(f"M2: uso de las cabezas (dominante: {int(uso.argmax())}, {100 * uso.max() / uso.sum():.1f} %)")
    fig.tight_layout(); fig.savefig(sal / "m2_cabezas.png", dpi=130); plt.close(fig)
    print("\n".join(md[:30]))
    print("\nListo ->", sal)


if __name__ == "__main__":
    main()
