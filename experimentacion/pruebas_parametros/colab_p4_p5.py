# @title 13. Pruebas P4 y P5 y ARI/NMI de la rejilla de P2 (necesitan la GPU)
# Pega esta celda en el cuaderno colab_2_m3_m4.ipynb DESPUES de la celda 9 (usa las variables de las celdas 1 a 7:
# WORK, NPY, REPO_DIR, PYBIN, ENV2, LOGS, OUT_DRIVE). Antes haz `git pull` del repo para traer las opciones --parche.
# Cada condicion cambia UN parametro de extraer_features.py sobre la misma matriz S de la capilla y compara el
# descriptor contra la referencia (parche 1024, precision media, rejilla de 2 cm) con k-means de 8 grupos (ARI y NMI).
import csv, os, re, shutil, subprocess, time
import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

PRUEBAS = f"{WORK}/pruebas"; os.makedirs(PRUEBAS, exist_ok=True)

def extraer(nombre, extra):
    salida = f"{PRUEBAS}/{nombre}.npz"
    cmd = [PYBIN, "extraer_features.py", "--input", NPY, "--output", salida, "--semilla", "0"] + extra
    r = subprocess.run(cmd, cwd=REPO_DIR, env=ENV2, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    open(f"{LOGS}/prueba_{nombre}.log", "w").write(r.stdout)
    if r.returncode != 0:
        print(f"❌ {nombre} fallo (puede ser falta de memoria):", r.stdout[-400:])
        return dict(nombre=nombre, archivo=None, vram_gb=None, forward_s=None, mb=None, error="fallo")
    vram = re.search(r"VRAM PICO:\s*([\d.]+)", r.stdout)
    seg = re.search(r"forward en ([\d.]+)s", r.stdout)
    return dict(nombre=nombre, archivo=salida, vram_gb=float(vram.group(1)) if vram else None,
                forward_s=float(seg.group(1)) if seg else None, mb=round(os.path.getsize(salida) / 2**20), error="")

def comparar(ref, cond):
    """ARI y NMI entre los k-means de dos descriptores, sobre los puntos que ambos comparten (a 5 cm o menos)."""
    a, b = np.load(ref["archivo"]), np.load(cond["archivo"])
    ca, cb = a["coord"], b["coord"]
    if ca.shape == cb.shape and np.allclose(ca, cb):
        ia = ib = np.arange(len(ca))
    else:
        d, j = cKDTree(ca).query(cb, k=1)
        ok = d <= 0.05
        ib, ia = np.where(ok)[0], j[ok]
    rng = np.random.default_rng(0)
    sel = rng.choice(len(ia), size=min(100_000, len(ia)), replace=False)
    fa, fb = a["feat"][ia[sel]].astype(np.float32), b["feat"][ib[sel]].astype(np.float32)
    fa = (fa - fa.mean(0)) / (np.linalg.norm(fa - fa.mean(0), axis=1, keepdims=True) + 1e-8)
    fb = (fb - fb.mean(0)) / (np.linalg.norm(fb - fb.mean(0), axis=1, keepdims=True) + 1e-8)
    la = KMeans(8, n_init=1, max_iter=30, random_state=0).fit_predict(fa)
    lb = KMeans(8, n_init=1, max_iter=30, random_state=0).fit_predict(fb)
    return adjusted_rand_score(la, lb), normalized_mutual_info_score(la, lb), len(ia)

ref = extraer("ref", [])
assert ref["archivo"], "La referencia no corrio"
CONDICIONES = [("P4", "precision simple (fp32)", "fp32", ["--fp32"]),
               ("P5", "parche 512", "parche512", ["--parche", "512"]),
               ("P5", "parche 2048", "parche2048", ["--parche", "2048"]),
               ("P2", "rejilla 4 cm", "grid004", ["--grid", "0.04"])]
filas = [dict(prueba="ref", condicion="referencia (parche 1024, media, 2 cm)", vram_gb=ref["vram_gb"], forward_s=ref["forward_s"],
              mb=ref["mb"], ari_vs_ref="", nmi_vs_ref="", puntos_comparados="", error="")]
for prueba, texto, nombre, extra in CONDICIONES:
    c = extraer(nombre, extra)
    if c["archivo"]:
        ari, nmi, n = comparar(ref, c)
        filas.append(dict(prueba=prueba, condicion=texto, vram_gb=c["vram_gb"], forward_s=c["forward_s"], mb=c["mb"],
                          ari_vs_ref=round(ari, 4), nmi_vs_ref=round(nmi, 4), puntos_comparados=n, error=""))
        os.remove(c["archivo"])
    else:
        filas.append(dict(prueba=prueba, condicion=texto, vram_gb="", forward_s="", mb="", ari_vs_ref="", nmi_vs_ref="",
                          puntos_comparados="", error=c["error"]))

DEST = f"{OUT_DRIVE}/pruebas_parametros"; os.makedirs(DEST, exist_ok=True)
with open(f"{DEST}/P4_P5_P2grid.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(filas[0].keys())); w.writeheader(); w.writerows(filas)
print("\nprueba | condicion | VRAM pico GB | forward s | MB | ARI | NMI | puntos")
for f in filas:
    print(f"{f['prueba']:4s} | {f['condicion']:38s} | {f['vram_gb']} | {f['forward_s']} | {f['mb']} | {f['ari_vs_ref']} | {f['nmi_vs_ref']} | {f['puntos_comparados']} {f['error']}")
print("\n💾 Guardado en", f"{DEST}/P4_P5_P2grid.csv", "(copialo a experimentacion/pruebas_parametros/resultados/)")
