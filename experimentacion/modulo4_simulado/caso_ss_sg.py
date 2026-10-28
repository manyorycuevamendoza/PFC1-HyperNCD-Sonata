import sys
sys.path.insert(0, "experimentacion/modulo4_simulado")
import numpy as np
import simular_modulo4 as s
from sklearn.cluster import DBSCAN

rng = np.random.default_rng(0)
xyz, cls, ins = s.escena(rng)
f_sem, f_ssl = s.descriptores(rng, cls, ins, sep_ssl=1.5)
nov = np.flatnonzero(cls >= 2)
reg = DBSCAN(eps=0.5, min_samples=5).fit_predict(xyz[nov])
ids = [r for r in np.unique(reg) if r >= 0]
clase_reg = {r: s.CLASES[np.bincount(cls[nov][reg == r]).argmax()] for r in ids}
elige = {}
for r in ids:
    elige.setdefault(clase_reg[r], r)
A, B, C = elige["columna"], elige["pilar"], elige["boveda"]
g = np.stack([s.rasgos_geometricos(xyz[nov][reg == r]) for r in ids])
gm, gs = g.mean(0), g.std(0) + 1e-9
def G(r):
    return (s.rasgos_geometricos(xyz[nov][reg == r]) - gm) / gs
def Sg(a, b):
    d2 = ((G(a) - G(b)) ** 2).sum()
    return np.exp(-d2 / 6)
def F(r, fx):
    m = nov[reg == r]
    return np.hstack([s.normalizar(f_sem[m]), s.normalizar(fx[m])]).mean(0)
def Ss(a, b, fx):
    fa, fb = F(a, fx), F(b, fx)
    return fa @ fb / np.linalg.norm(fa) / np.linalg.norm(fb)
geo = {r: s.rasgos_geometricos(xyz[nov][reg == r]) for r in ids}
f_geo = np.zeros((len(xyz), 6), np.float32)
for r in ids:
    f_geo[nov[reg == r]] = geo[r]
for (n1, a), (n2, b) in ((("columna", A), ("pilar", B)), (("columna", A), ("boveda", C))):
    print(f"{n1} vs {n2}:  S_g {Sg(a, b):.2f}   S_s base {Ss(a, b, f_geo):.2f}   "
          f"S_s propuesta {Ss(a, b, f_ssl):.2f}")
print("rasgos columna", np.round(geo[A][:3], 2), "pilar", np.round(geo[B][:3], 2),
      "boveda", np.round(geo[C][:3], 2))
