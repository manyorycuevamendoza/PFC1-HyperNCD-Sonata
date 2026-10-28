"""Simulacion controlada del Modulo 4 (fusion, hipergrafo y pseudo-etiquetas).

Construye una escena sintetica de una nave de iglesia en la que se conoce la
clase de cada punto, le inventa los descriptores que entregarian los Modulos 2
y 3 con una calidad controlada, y ejecuta sobre ella las formulas del Modulo 4
del Capitulo 2:

    F_i = [f_sem ; f_x]                 f_x = f_geo (linea base) o f_ssl (propuesta)
    regiones   DBSCAN sobre las coordenadas de los puntos novedosos
    S_s        coseno entre descriptores de region             (Ec. coseno)
    S_g        nucleo gaussiano entre rasgos geometricos g     (Ec. gauss)
    S_b        = alpha S_g + (1 - alpha) S_s                   (Ec. afinidad)
    hiperarista  cada region con sus M regiones mas afines; su descriptor es el
                 promedio ponderado por la afinidad
    prototipos   promedio ponderado por la pertenencia s_ik    (Ec. prototipo)
    pseudo-etiquetas  Sinkhorn-Knopp semi-relajado              (Ec. sk-costo, sk-iter)

Lo que NO es: no entrena ninguna red. En HyperNCD las cabezas y la red del
Modulo 2 se ajustan con la perdida epoca a epoca; aqui los prototipos se
refinan alternando asignacion y promedio (un EM), que es lo que haria ese ciclo
con los descriptores fijos. Sirve para ver el comportamiento del Modulo 4 y
como depende de cada parametro, no para predecir el mIoU real de la Catedral.

Uso:
    python experimentacion/modulo4_simulado/simular_modulo4.py --salida experimentacion/modulo4_simulado/resultados
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score

CONOCIDAS = ["suelo", "muro"]
NOVEDOSAS = ["columna", "pilar", "boveda"]
CLASES = CONOCIDAS + NOVEDOSAS


# --------------------------------------------------------------------------
# 1. Escena sintetica con verdad de terreno
# --------------------------------------------------------------------------
def escena(rng, densidad=1.0, ruido_xyz=0.01, n_pilar=1.0):
    """Nave de 20 x 10 m: suelo y muros (conocidas), columnas, pilares y boveda
    (novedosas). Columna y pilar tienen la MISMA forma, un cilindro de 0,3 m de
    radio y 6 m de alto, y estan intercalados en la nave: la geometria y la
    posicion no los distinguen, solo el significado. Es el caso que justifica
    sustituir f_geo por f_ssl. Devuelve coordenadas (N,3), clase (N,) e
    instancia (N,)."""
    xyz, cls, ins = [], [], []
    k = 0

    def agregar(p, c):
        nonlocal k
        xyz.append(p)
        cls.append(np.full(len(p), CLASES.index(c)))
        ins.append(np.full(len(p), k))
        k += 1

    n = lambda m: max(10, int(m * densidad))
    # suelo
    p = np.c_[rng.uniform(0, 20, n(6000)), rng.uniform(0, 10, n(6000)), np.zeros(n(6000))]
    agregar(p, "suelo")
    # dos muros laterales
    for y in (0.0, 10.0):
        m = n(3000)
        agregar(np.c_[rng.uniform(0, 20, m), np.full(m, y), rng.uniform(0, 6, m)], "muro")
    # ocho cilindros de 0,3 m de radio y 6 m de alto, columnas y pilares
    # intercalados en damero para que la posicion no delate la clase
    for i, x in enumerate(np.linspace(2, 18, 4)):
        for j, y in enumerate((2.0, 8.0)):
            clase = "columna" if (i + j) % 2 == 0 else "pilar"
            m = n(700 * (n_pilar if clase == "pilar" else 1.0))
            a = rng.uniform(0, 2 * np.pi, m)
            agregar(np.c_[x + 0.3 * np.cos(a), y + 0.3 * np.sin(a), rng.uniform(0, 6, m)],
                    clase)
    # boveda de canon en cuatro tramos, semicilindro sobre los muros
    for x0 in (0.0, 5.0, 10.0, 15.0):
        m = n(1500)
        a = rng.uniform(0, np.pi, m)
        agregar(np.c_[rng.uniform(x0 + 0.5, x0 + 4.5, m), 5 + 5 * np.cos(a), 6 + 3 * np.sin(a)],
                "boveda")
    xyz = np.concatenate(xyz)
    xyz += rng.normal(0, ruido_xyz, xyz.shape)
    return xyz, np.concatenate(cls), np.concatenate(ins)


# --------------------------------------------------------------------------
# 2. Descriptores simulados de los Modulos 2 y 3
# --------------------------------------------------------------------------
def unitarios(rng, k, d):
    v = rng.normal(size=(k, d))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def descriptores(rng, cls, ins, sep_sem_nov=0.3, sep_ssl=2.0, var_obj=1.0,
                 ruido=1.0, d_sem=96, d_ssl=1088):
    """Descriptor = centro de su clase + desvio propio de su objeto + ruido.

    f_sem separa bien las conocidas y apenas las novedosas, porque la red solo
    vio etiquetas de las conocidas (sep_sem_nov). f_ssl separa las novedosas con
    una fuerza controlada (sep_ssl), que hace de 'calidad de Sonata'.
    El desvio por objeto (var_obj) es lo que hace dificil la tarea: cada
    columna o pilar concreto se aparta de su clase, y ese desvio no se cancela
    al promediar los puntos de una region, como si lo hace el ruido por punto."""
    n = len(cls)
    n_obj = ins.max() + 1
    c_sem = unitarios(rng, len(CLASES), d_sem) * 3.0
    comun = unitarios(rng, 1, d_sem)[0] * 3.0            # las novedosas se parecen
    for c in range(len(CONOCIDAS), len(CLASES)):
        c_sem[c] = comun + (c_sem[c] / 3.0) * sep_sem_nov * 3.0
    f_sem = (c_sem[cls] + unitarios(rng, n_obj, d_sem)[ins] * var_obj * 1.5
             + rng.normal(0, ruido, (n, d_sem)))
    c_ssl = unitarios(rng, len(CLASES), d_ssl) * sep_ssl
    f_ssl = (c_ssl[cls] + unitarios(rng, n_obj, d_ssl)[ins] * var_obj
             + rng.normal(0, ruido / np.sqrt(d_ssl / d_sem), (n, d_ssl)))
    return f_sem.astype(np.float32), f_ssl.astype(np.float32)


def rasgos_geometricos(xyz):
    """f_geo de una region: linealidad, planaridad, dispersion y centroide."""
    c = xyz.mean(0)
    if len(xyz) < 4:
        return np.r_[0, 0, 1, c]
    ev = np.sort(np.linalg.eigvalsh(np.cov((xyz - c).T)))[::-1] + 1e-12
    l1, l2, l3 = ev
    return np.r_[(l1 - l2) / l1, (l2 - l3) / l1, l3 / l1, c]


def normalizar(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


# --------------------------------------------------------------------------
# 3. Modulo 4
# --------------------------------------------------------------------------
def sinkhorn(o, gamma=1.0, eps_sk=0.05, iters=3, peso=None):
    """Ec. sk-costo y sk-iter: Q = softmax(o/0,1)^(1/eps), escalado semi-relajado.
    peso: cuantos puntos representa cada fila. La tesis equilibra puntos, no
    regiones; una region de n puntos cuenta como n filas iguales."""
    B, K = o.shape
    peso = np.ones(B) if peso is None else np.asarray(peso, float)
    z = o / 0.1
    z = z - z.max(1, keepdims=True)
    q = np.exp(z)
    q /= q.sum(1, keepdims=True)
    Q = q ** (1 / eps_sk) + 1e-30
    Q /= Q.sum()
    r = peso / peso.sum()
    u, v = np.ones(B), np.ones(K)
    for _ in range(iters):
        u = r / (Q @ v)
        v = ((1 / K) / (Q.T @ u)) ** (gamma / (gamma + eps_sk))
    Y = u[:, None] * Q * v[None, :]
    return Y / (Y.sum(1, keepdims=True) + 1e-30)          # reparto de cada fila


def modulo4(xyz, f_sem, f_x, nov, rng, alpha=0.5, M=8, tau=1.0, eps_dbscan=0.5,
            gamma=1.0, iters_em=15, pesar_ramas=True):
    """Devuelve la etiqueta de cada punto novedoso (indices de nov) y el numero
    de regiones."""
    p = xyz[nov]
    reg = DBSCAN(eps=eps_dbscan, min_samples=5).fit_predict(p)
    ids = np.unique(reg[reg >= 0])
    if len(ids) < len(NOVEDOSAS):
        return np.full(len(nov), -1), len(ids)
    # descriptor F de cada punto, ramas al mismo peso antes de concatenar
    a, b = f_sem[nov], f_x[nov]
    if pesar_ramas:
        a, b = normalizar(a), normalizar(b)
    F = np.hstack([a, b])
    Freg = np.stack([F[reg == r].mean(0) for r in ids])
    g = np.stack([rasgos_geometricos(p[reg == r]) for r in ids])
    g = (g - g.mean(0)) / (g.std(0) + 1e-9)
    # afinidad entre regiones
    Fn = normalizar(Freg)
    Ss = Fn @ Fn.T
    d2 = ((g[:, None] - g[None]) ** 2).sum(-1)
    Sg = np.exp(-d2 / (tau * g.shape[1]))
    Sb = alpha * Sg + (1 - alpha) * Ss
    # hiperaristas: region y sus M mas afines, promedio ponderado por la afinidad
    m = min(M, len(ids) - 1)
    vecinos = np.argsort(-Sb, axis=1)[:, : m + 1]          # incluye a la propia region
    w = np.take_along_axis(np.clip(Sb, 0, None), vecinos, 1)
    Fh = (w[..., None] * Freg[vecinos]).sum(1) / (w.sum(1, keepdims=True) + 1e-12)
    Fh = normalizar(Fh)
    # prototipos: k-means++ en coseno y refinamiento con Sinkhorn (EM)
    K = len(NOVEDOSAS)
    P = [Fh[rng.integers(len(Fh))]]
    for _ in range(K - 1):
        d = 1 - (Fh @ np.stack(P).T).max(1)
        P.append(Fh[rng.choice(len(Fh), p=d / d.sum())])
    P = np.stack(P)
    tam = np.array([(reg == r).sum() for r in ids])         # puntos por region
    for _ in range(iters_em):
        s = sinkhorn(Fh @ normalizar(P).T, gamma=gamma, peso=tam)   # pertenencia s_ik
        sw = s * tam[:, None]                                 # cada punto pesa igual
        P = (sw.T @ Fh) / (sw.sum(0)[:, None] + 1e-12)        # Ec. prototipo
    Y = s
    et_reg = Y.argmax(1)
    et = np.full(len(nov), -1)
    for r, e in zip(ids, et_reg):
        et[reg == r] = e
    return et, len(ids)


# --------------------------------------------------------------------------
# 4. Metricas
# --------------------------------------------------------------------------
def miou_novedosas(verdad, pred):
    """mIoU de las clases novedosas con emparejamiento hungaro (las clases
    descubiertas no tienen nombre). Los puntos sin region cuentan como error."""
    K = len(NOVEDOSAS)
    ok = pred >= 0
    conf = np.zeros((K, K))
    np.add.at(conf, (verdad[ok], pred[ok]), 1)
    fila, col = linear_sum_assignment(-conf)
    iou = []
    for v, q in zip(fila, col):
        inter = conf[v, q]
        union = (verdad == v).sum() + (pred == q).sum() - inter
        iou.append(inter / union)
    return float(np.mean(iou)), [float(x) for x in iou]


def correr(semilla, base, ssl, **kw):
    """Una escena, las dos versiones del descriptor (f_geo y f_ssl)."""
    rng = np.random.default_rng(semilla)
    xyz, cls, ins = escena(rng, densidad=kw.pop("densidad", 1.0),
                           ruido_xyz=kw.pop("ruido_xyz", 0.01),
                           n_pilar=kw.pop("n_pilar", 1.0))
    f_sem, f_ssl = descriptores(rng, cls, ins, sep_sem_nov=kw.pop("sep_sem_nov", 0.3),
                                sep_ssl=kw.pop("sep_ssl", 1.5),
                                var_obj=kw.pop("var_obj", 1.0))
    nov = np.flatnonzero(cls >= len(CONOCIDAS))
    verdad = cls[nov] - len(CONOCIDAS)
    # f_geo por punto: el de su region, como en la implementacion ejecutada
    reg = DBSCAN(eps=kw.get("eps_dbscan", 0.5), min_samples=5).fit_predict(xyz[nov])
    f_geo = np.zeros((len(xyz), 6), np.float32)
    for r in np.unique(reg[reg >= 0]):
        f_geo[nov[reg == r]] = rasgos_geometricos(xyz[nov][reg == r])
    out = {}
    for nombre, fx in (("base", f_geo), ("propuesta", f_ssl)):
        pred, nreg = modulo4(xyz, f_sem, fx, nov, np.random.default_rng(semilla), **kw)
        m, iou = miou_novedosas(verdad, pred)
        out[nombre] = dict(miou=m, iou=iou, ari=float(adjusted_rand_score(verdad, pred)),
                           regiones=nreg, pred=pred)
    out.update(xyz=xyz, cls=cls, nov=nov)
    return out


ESCENARIOS = {
    "calidad_sonata": ("sep_ssl", [0.0, 0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0]),
    "alpha":          ("alpha",   [0.0, 0.25, 0.5, 0.75, 1.0]),
    "M":              ("M",       [1, 2, 4, 8, 16]),
    "desbalance":     ("n_pilar", [1.0, 0.5, 0.25, 0.1]),
    "ruido_xyz":      ("ruido_xyz", [0.005, 0.02, 0.05, 0.1]),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--salida", default="experimentacion/modulo4_simulado/resultados")
    ap.add_argument("--semillas", type=int, default=5, help="repeticiones por condicion")
    args = ap.parse_args()
    sal = Path(args.salida)
    sal.mkdir(parents=True, exist_ok=True)

    filas = []
    for esc, (param, valores) in ESCENARIOS.items():
        for v in valores:
            res = [correr(s, None, None, **{param: v}) for s in range(args.semillas)]
            for version in ("base", "propuesta"):
                mi = [r[version]["miou"] for r in res]
                ar = [r[version]["ari"] for r in res]
                filas.append(dict(escenario=esc, parametro=param, valor=v, version=version,
                                  miou=float(np.mean(mi)), miou_std=float(np.std(mi)),
                                  ari=float(np.mean(ar)),
                                  regiones=float(np.mean([r[version]["regiones"] for r in res]))))
            print(f"{esc:15s} {param}={v:<6}  base mIoU {filas[-2]['miou']:.3f}  "
                  f"propuesta {filas[-1]['miou']:.3f}", flush=True)

    import csv
    with open(sal / "resultados.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)

    # escena de referencia para las figuras 3D
    ref = correr(0, None, None)
    np.savez_compressed(sal / "escena_referencia.npz", xyz=ref["xyz"], cls=ref["cls"],
                        nov=ref["nov"], pred_base=ref["base"]["pred"],
                        pred_propuesta=ref["propuesta"]["pred"])
    (sal / "referencia.json").write_text(json.dumps(
        {v: {k: ref[v][k] for k in ("miou", "iou", "ari", "regiones")}
         for v in ("base", "propuesta")}, indent=2))
    print(f"[ok] {sal}")


if __name__ == "__main__":
    main()
