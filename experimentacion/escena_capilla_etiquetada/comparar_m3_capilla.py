"""Medidas de f_ssl sobre la escena etiquetada, sin usar etiquetas para medir (solo para elegir los puntos).

Compara Sonata original y adaptado con las mismas medidas que `experimentacion/adaptacion_sonata/comparar_m3.py`
(hechas alli sobre PL13) y dibuja la similitud coseno y el k-means del descriptor original.
Solo se usan los puntos de las cinco piezas etiquetadas. La clase `otro` queda fuera porque confunde la lectura.

Uso (desde la raiz del repo):
    python experimentacion/escena_capilla_etiquetada/comparar_m3_capilla.py \
        --npy <capilla.npy> --etiquetas <etiquetas.npy> --original <feats_original.npz> --adaptado <feats_adaptado.npz> \
        --json <salida.json> --figura <salida.png>
"""
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

ap = argparse.ArgumentParser()
ap.add_argument("--npy", required=True)
ap.add_argument("--etiquetas", required=True)
ap.add_argument("--original", required=True)
ap.add_argument("--adaptado", required=True)
ap.add_argument("--json", required=True)
ap.add_argument("--figura", required=True)
ap.add_argument("--muestra", type=int, default=100000)
a = ap.parse_args()
rng = np.random.default_rng(0)

S = np.load(a.npy, mmap_mode="r")
y = np.load(a.etiquetas).astype(int)
lab = np.where(y < 5)[0]
xyz = np.asarray(S[lab, :3], dtype=np.float32)


def cargar(ruta):
    z = np.load(ruta)
    return z["feat"][z["inverse"][lab]].astype(np.float32)


F = {"original": cargar(a.original), "adaptado": cargar(a.adaptado)}
m = rng.choice(len(lab), size=min(a.muestra, len(lab)), replace=False)


def normalizar(X):
    X = X - X.mean(0)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


N = {k: normalizar(v[m]) for k, v in F.items()}
res = {"puntos_etiquetados": int(len(lab)), "muestra": int(len(m))}

cos = np.sum(N["original"] * N["adaptado"], axis=1)
res["coseno_original_adaptado"] = {"media": float(cos.mean()), "p5": float(np.percentile(cos, 5))}

for k, X in N.items():
    p = PCA(n_components=min(200, X.shape[1]), random_state=0).fit(X)
    v = p.explained_variance_ratio_
    pk = p.explained_variance_
    pk = pk / pk.sum()
    # rango efectivo con todas las componentes: se usa la covarianza completa
    ev = np.linalg.eigvalsh(np.cov(X.T))[::-1]
    ev = np.clip(ev, 0, None)
    q = ev / ev.sum()
    q = q[q > 0]
    res[k] = {"var_pca_3": float(v[:3].sum()), "var_pca_10": float(v[:10].sum()),
              "rango_efectivo": float(np.exp(-(q * np.log(q)).sum()))}

grupos = {}
for k, X in N.items():
    g = [KMeans(8, n_init=1, max_iter=30, random_state=s).fit(X).labels_ for s in range(3)]
    grupos[k] = g
    res[k]["estabilidad_kmeans"] = float(np.mean([adjusted_rand_score(g[i], g[j]) for i in range(3) for j in range(i + 1, 3)]))
res["ari_kmeans_original_adaptado"] = float(adjusted_rand_score(grupos["original"][0], grupos["adaptado"][0]))
json.dump(res, open(a.json, "w"), indent=2)
print(json.dumps(res, indent=2))

# figura con el descriptor original sobre todos los puntos etiquetados
X = normalizar(F["original"])
km = KMeans(8, n_init=1, max_iter=30, random_state=0).fit(X[m])
etiq = km.predict(X)
# punto de consulta, uno de la cupula (clase 2) cercano a su mediana
c2 = np.where(y[lab] == 2)[0]
q = c2[np.argmin(np.linalg.norm(xyz[c2] - np.median(xyz[c2], axis=0), axis=1))]
sim = X @ X[q]
ds = rng.choice(len(lab), size=min(250000, len(lab)), replace=False)
ds = ds[np.argsort(xyz[ds, 2])]
P = xyz[ds]
fig, ejes = plt.subplots(2, 2, figsize=(6.6, 7.4))
for j, (tit, col, cmap) in enumerate([("(a) Similitud coseno con un punto de la cúpula", sim[ds], "viridis"),
                                       ("(b) $k$-means, 8 grupos", etiq[ds], "tab10")]):
    for i, ejey in enumerate((1, 2)):
        sc = ejes[i, j].scatter(P[:, 0], P[:, ejey], c=col, s=0.3, linewidths=0, cmap=cmap, rasterized=True,
                                vmin=(np.percentile(sim, 2) if j == 0 else None), vmax=(np.percentile(sim, 99.5) if j == 0 else None))
        ejes[i, j].set_aspect("equal")
        ejes[i, j].set_xticks([])
        ejes[i, j].set_yticks([])
    ejes[0, j].set_title(tit, fontsize=9)
ejes[0, 0].set_ylabel("planta (x, y)")
ejes[1, 0].set_ylabel("lateral (x, z)")
fig.subplots_adjust(top=0.93, bottom=0.02, wspace=0.04, hspace=0.04, left=0.06, right=0.99)
fig.savefig(a.figura, dpi=130)
