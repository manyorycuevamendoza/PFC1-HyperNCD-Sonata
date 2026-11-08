"""Compara el f_ssl de Sonata ORIGINAL contra el ADAPTADO a la Catedral (Modulo 3), sobre PL13.

Los dos .npz salen de `extraer_features.py` (con y sin `--pesos`). No tienen el mismo
numero de filas en la rejilla (796 315 contra 796 309), asi que se comparan POR PUNTO
ORIGINAL usando `inverse`, no por indice de fila.

Mide, sin etiquetas:
  - similitud coseno entre el descriptor original y el adaptado del mismo punto
    (cruda y centrada; la centrada quita la componente comun a toda la escena);
  - varianza de la PCA y rango efectivo de cada uno (que tan "rico" es el espacio);
  - k-means con K grupos en cada uno y su acuerdo (ARI) y estabilidad entre semillas.
Dibuja PCA-RGB, k-means y un mapa de cuanto cambio cada punto.

Uso (desde la raiz del repo):
    python experimentacion/adaptacion_sonata/comparar_m3.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score

RAIZ = Path(__file__).resolve().parents[2]
DATOS = RAIZ / "experimentacion" / "datos"


def args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--original", default=str(DATOS / "modulo3_PL13" / "PL13_feats.npz"))
    ap.add_argument("--adaptado", default=str(DATOS / "modulo3_PL13" / "PL13_feats_adaptado.npz"))
    ap.add_argument("--pl7-adaptado", default=str(DATOS / "modulo3_PL7" / "PL7_feats_adaptado.npz"))
    ap.add_argument("--npy", default=str(DATOS / "modulo3_PL13" / "PL13_SCN0001.npy"))
    ap.add_argument("--salida", default=str(Path(__file__).parent / "resultados" / "m3"))
    ap.add_argument("--puntos", type=int, default=200_000, help="puntos originales a comparar")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--semilla", type=int, default=0)
    return ap.parse_args()


def pca_info(f, rng, n=100_000):
    """Varianza explicada (3 y 10 comps), rango efectivo y proyeccion a 3 comps."""
    idx = rng.choice(len(f), min(len(f), n), replace=False)
    media = f[idx].mean(0)
    _, s, vt = np.linalg.svd(f[idx] - media, full_matrices=False)
    var = s ** 2 / (s ** 2).sum()
    p = var[var > 0]
    return {
        "var_pca3": float(var[:3].sum()), "var_pca10": float(var[:10].sum()),
        "rango_efectivo": float(np.exp(-(p * np.log(p)).sum())),
    }, media, vt[:3]


def rgb(x):
    lo, hi = np.percentile(x, [2, 98], axis=0)
    return np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)


def kmeans_estable(x, k, semillas=(0, 1, 2)):
    etiquetas = [MiniBatchKMeans(k, random_state=s, n_init=3, batch_size=8192).fit_predict(x) for s in semillas]
    pares = [adjusted_rand_score(etiquetas[i], etiquetas[j]) for i in range(len(etiquetas)) for j in range(i + 1, len(etiquetas))]
    return etiquetas[0], float(np.mean(pares))


def main():
    a = args()
    rng = np.random.default_rng(a.semilla)
    salida = Path(a.salida)
    salida.mkdir(parents=True, exist_ok=True)

    zo, za = np.load(a.original), np.load(a.adaptado)
    S = np.load(a.npy, mmap_mode="r")
    n = len(zo["inverse"])
    assert n == len(za["inverse"]) == len(S), "los .npz no corresponden a este .npy"
    sel = np.sort(rng.choice(n, min(n, a.puntos), replace=False))      # puntos ORIGINALES
    fo = zo["feat"][zo["inverse"][sel]].astype(np.float32)
    fa = za["feat"][za["inverse"][sel]].astype(np.float32)
    xyz = np.asarray(S[sel, :3])
    print(f"PL13: {n:,} puntos originales; se comparan {len(sel):,}. "
          f"Filas de rejilla: original {len(zo['feat']):,}, adaptado {len(za['feat']):,}")

    # --- cuanto cambio cada punto ---
    cos = (fo * fa).sum(1) / (np.linalg.norm(fo, axis=1) * np.linalg.norm(fa, axis=1) + 1e-8)
    po, mo, vo = pca_info(zo["feat"].astype(np.float32), rng)
    pa, ma, va = pca_info(za["feat"].astype(np.float32), rng)
    co, ca = fo - mo, fa - ma
    cos_c = (co * ca).sum(1) / (np.linalg.norm(co, axis=1) * np.linalg.norm(ca, axis=1) + 1e-8)
    dif_rel = np.linalg.norm(fa - fo, axis=1) / (np.linalg.norm(fo, axis=1) + 1e-8)

    # --- k-means en cada espacio y su acuerdo ---
    no, na = co / (np.linalg.norm(co, axis=1, keepdims=True) + 1e-8), ca / (np.linalg.norm(ca, axis=1, keepdims=True) + 1e-8)
    lo, est_o = kmeans_estable(no, a.k)
    la, est_a = kmeans_estable(na, a.k)
    ari = float(adjusted_rand_score(lo, la))

    res = {
        "puntos_comparados": int(len(sel)), "puntos_originales": int(n),
        "filas_rejilla": {"original": int(len(zo["feat"])), "adaptado": int(len(za["feat"]))},
        "coseno_crudo": {"media": float(cos.mean()), "p5": float(np.percentile(cos, 5)), "min": float(cos.min())},
        "coseno_centrado": {"media": float(cos_c.mean()), "p5": float(np.percentile(cos_c, 5)), "min": float(cos_c.min())},
        "cambio_relativo_L2": {"media": float(dif_rel.mean()), "p95": float(np.percentile(dif_rel, 95))},
        "original": {**po, "kmeans_estabilidad_ari": est_o, "kmeans_tamanos": np.bincount(lo, minlength=a.k).tolist()},
        "adaptado": {**pa, "kmeans_estabilidad_ari": est_a, "kmeans_tamanos": np.bincount(la, minlength=a.k).tolist()},
        "kmeans_K": a.k, "acuerdo_ari_original_vs_adaptado": ari,
    }

    # --- PL7 (solo hay adaptado; no hay original para comparar) ---
    try:
        z7 = np.load(a.pl7_adaptado)
        p7, _, _ = pca_info(z7["feat"].astype(np.float32), rng)
        res["PL7_adaptado_solo"] = {**p7, "filas_rejilla": int(len(z7["feat"])), "dim": int(z7["feat"].shape[1])}
    except FileNotFoundError:
        pass

    (salida / "comparacion_m3_PL13.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False))

    # --- figura ---
    pc_o, pc_a = rgb(co @ vo.T), rgb(ca @ va.T)
    # cada PCA tiene su propio signo/orden arbitrario; los colores NO son comparables entre paneles
    fig, ejes = plt.subplots(2, 3, figsize=(18, 11))
    x, z = xyz[:, 0], xyz[:, 1]
    def paneles(ax, c, t, **kw):
        ax.scatter(x, z, c=c, s=0.4, linewidths=0, rasterized=True, **kw)
        ax.set_title(t, fontsize=11); ax.set_aspect("equal"); ax.axis("off")
    paneles(ejes[0, 0], pc_o, "PCA de f_ssl ORIGINAL (RGB)")
    paneles(ejes[0, 1], pc_a, "PCA de f_ssl ADAPTADO (RGB)")
    paneles(ejes[0, 2], cos_c, f"Coseno centrado orig vs adapt (media {cos_c.mean():.2f})", cmap="viridis", vmin=0, vmax=1)
    paneles(ejes[1, 0], lo, f"k-means K={a.k} ORIGINAL (estab. {est_o:.2f})", cmap="tab10")
    paneles(ejes[1, 1], la, f"k-means K={a.k} ADAPTADO (estab. {est_a:.2f})", cmap="tab10")
    ejes[1, 2].hist(cos_c, bins=80, color="#2b7bba")
    ejes[1, 2].set_title(f"Distribucion del coseno centrado  |  ARI k-means orig~adapt = {ari:.2f}")
    ejes[1, 2].set_xlabel("coseno (1 = el punto no cambio)")
    fig.suptitle("PL13: Sonata original vs adaptado a la Catedral (vista desde arriba, X-Y). "
                 "Los colores de PCA no son comparables entre paneles.", fontsize=12)
    fig.tight_layout()
    fig.savefig(salida / "comparacion_m3_PL13.png", dpi=130)
    print("figura ->", salida / "comparacion_m3_PL13.png")


if __name__ == "__main__":
    main()
