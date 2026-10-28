"""Una figura por paso del Modulo 4 simulado, sobre la escena de referencia
(repeticion 1, calidad de Sonata 1,5), para documentar cada paso en la tesis.

    paso2_regiones.png    regiones de DBSCAN sobre los puntos de clases nuevas
    paso3_afinidad.png    matrices S_g, S_s de la linea base y S_s de la propuesta
    paso4_hiperarista.png una region y sus M regiones mas afines, en las dos versiones

Uso:
    python experimentacion/modulo4_simulado/figuras_pasos.py --dir experimentacion/modulo4_simulado/resultados
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

sys.path.insert(0, str(Path(__file__).parent))
import simular_modulo4 as s  # noqa: E402


def preparar(alpha=0.5, M=8):
    rng = np.random.default_rng(0)
    xyz, cls, ins = s.escena(rng)
    f_sem, f_ssl = s.descriptores(rng, cls, ins, sep_ssl=1.5)
    nov = np.flatnonzero(cls >= 2)
    p = xyz[nov]
    reg = DBSCAN(eps=0.5, min_samples=5).fit_predict(p)
    ids = np.unique(reg[reg >= 0])
    clase = np.array([np.bincount(cls[nov][reg == r]).argmax() - 2 for r in ids])
    orden = np.lexsort((ids, clase))                       # regiones agrupadas por clase
    ids, clase = ids[orden], clase[orden]
    g = np.stack([s.rasgos_geometricos(p[reg == r]) for r in ids])
    f_geo = np.zeros((len(xyz), 6), np.float32)
    for r, gi in zip(ids, g):
        f_geo[nov[reg == r]] = gi
    gz = (g - g.mean(0)) / (g.std(0) + 1e-9)
    Sg = np.exp(-((gz[:, None] - gz[None]) ** 2).sum(-1) / gz.shape[1])
    out = dict(xyz=xyz, cls=cls, nov=nov, reg=reg, ids=ids, clase=clase, Sg=Sg)
    for nombre, fx in (("base", f_geo), ("propuesta", f_ssl)):
        F = np.hstack([s.normalizar(f_sem[nov]), s.normalizar(fx[nov])])
        Fr = s.normalizar(np.stack([F[reg == r].mean(0) for r in ids]))
        Ss = Fr @ Fr.T
        Sb = alpha * Sg + (1 - alpha) * Ss
        out[nombre] = dict(Ss=Ss, Sb=Sb,
                           vecinos=np.argsort(-Sb, axis=1)[:, : min(M, len(ids) - 1) + 1])
    return out


def regiones(o, d):
    xyz, nov, reg = o["xyz"], o["nov"], o["reg"]
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(projection="3d")
    cmap = plt.get_cmap("tab20")
    col = np.where(reg[:, None] >= 0, cmap(reg % 20)[:, :3], [[0, 0, 0]])
    ax.scatter(*xyz[nov].T, c=col, s=0.6, linewidths=0)
    ax.set_title(f"{len(o['ids'])} regiones de DBSCAN, radio 0,5 m", fontsize=12)
    ax.view_init(elev=28, azim=-62)
    ax.set_box_aspect(np.ptp(xyz, axis=0))
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(d / "paso2_regiones.png", dpi=150)


def afinidad(o, d):
    nombres = ["columna", "pilar", "bóveda"]
    mats = [("$S_g$, forma y posición", o["Sg"]),
            ("$S_s$ línea base, con $f_{geo}$", o["base"]["Ss"]),
            ("$S_s$ propuesta, con $f_{ssl}$", o["propuesta"]["Ss"])]
    fig, ejes = plt.subplots(1, 3, figsize=(15, 4.8))
    cortes = np.flatnonzero(np.diff(o["clase"])) + 0.5
    centros = [np.mean(np.flatnonzero(o["clase"] == k)) for k in range(3)]
    for ax, (tit, m) in zip(ejes, mats):
        im = ax.imshow(m, vmin=0, vmax=1, cmap="viridis")
        for c in cortes:
            ax.axhline(c, color="w", lw=1.5)
            ax.axvline(c, color="w", lw=1.5)
        ax.set_xticks(centros, nombres)
        ax.set_yticks(centros, nombres)
        ax.set_title(tit, fontsize=12)
    fig.colorbar(im, ax=ejes, shrink=0.8, label="afinidad")
    fig.savefig(d / "paso3_afinidad.png", dpi=150, bbox_inches="tight")


def hiperarista(o, d):
    xyz, nov, reg, ids, clase = o["xyz"], o["nov"], o["reg"], o["ids"], o["clase"]
    i = int(np.flatnonzero(clase == 0)[0])                 # una region de columna
    fig = plt.figure(figsize=(12, 5))
    for k, ver in enumerate(("base", "propuesta"), 1):
        vec = o[ver]["vecinos"][i]
        ax = fig.add_subplot(1, 2, k, projection="3d")
        col = np.tile([[0.85, 0.85, 0.85]], (len(nov), 1))
        for j in vec[1:]:
            col[reg == ids[j]] = [0.16, 0.50, 0.73]
        col[reg == ids[i]] = [0.90, 0.20, 0.10]
        ax.scatter(*xyz[nov].T, c=col, s=0.6, linewidths=0)
        bien = int((clase[vec[1:]] == 0).sum())
        ax.set_title(f"{'Línea base' if ver == 'base' else 'Propuesta'}, "
                     f"{bien} de {len(vec) - 1} vecinos son columnas", fontsize=12)
        ax.view_init(elev=28, azim=-62)
        ax.set_box_aspect(np.ptp(xyz, axis=0))
        ax.set_axis_off()
    fig.suptitle("Hiperarista de una columna (rojo) y sus regiones más afines (azul)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(d / "paso4_hiperarista.png", dpi=150)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="experimentacion/modulo4_simulado/resultados")
    d = Path(ap.parse_args().dir)
    o = preparar()
    regiones(o, d)
    afinidad(o, d)
    hiperarista(o, d)
    for ver in ("base", "propuesta"):
        vec = o[ver]["vecinos"][int(np.flatnonzero(o["clase"] == 0)[0])][1:]
        print(ver, "vecinos de la primera columna, por clase:", list(o["clase"][vec]))
    print("regiones por clase:", np.bincount(o["clase"]))
    print(f"[ok] {d}")


if __name__ == "__main__":
    main()
