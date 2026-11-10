# @title M4 en la capilla con el M2 entrenado en ArCH (CPU, sin Sonata: usa los f_ssl que ya están en Drive)
# Pegar en una celda de Colab (cualquier runtime, no hace falta GPU). Antes: haber hecho `git push` de
# experimentacion/modulo4_real/correr_modulo4_real.py (tiene la opción nueva --todos).
import os, glob, shutil, subprocess, sys, zipfile
from google.colab import drive, userdata
if not os.path.exists("/content/drive/MyDrive"):
    drive.mount("/content/drive")

DRIVE = "/content/drive/MyDrive/PFC1_HyperNCD_Sonata"
ESCENA = "capilla_etiquetada"
FSEM_ARCH = f"{DRIVE}/arch/entrenamiento/predicciones/capilla_etiquetada_last_fsem.npz"   # o ..._best_fsem.npz
NOMBRE = "arch_e30"                              # identifica la corrida
K_LISTA = [6, 8, 12]
REPO = "/content/PFC1-HyperNCD-Sonata"
WORK = "/content/work_m4_arch"; os.makedirs(WORK, exist_ok=True)
SAL = f"{DRIVE}/arch/entrenamiento/m4_capilla_{NOMBRE}"; os.makedirs(SAL, exist_ok=True)

# repo al día (token desde Secrets, como en los otros cuadernos)
tok = userdata.get("GITHUB_TOKEN")
if not os.path.exists(REPO):
    subprocess.run(f"git clone https://{tok}@github.com/manyorycuevamendoza/PFC1-HyperNCD-Sonata.git {REPO}", shell=True, check=True)
else:
    subprocess.run(f"cd {REPO} && git pull origin main", shell=True, check=True)
src = open(f"{REPO}/experimentacion/modulo4_real/correr_modulo4_real.py", encoding="utf-8").read()
assert "--todos" in src, "Falta hacer push de correr_modulo4_real.py (opción --todos)."

# entradas (todas ya están en Drive)
M1 = f"{DRIVE}/m1/salida/{ESCENA}"
for n in (f"{ESCENA}.npy", f"{ESCENA}_etiquetas.npy", f"{ESCENA}_etiquetas.json"):
    shutil.copy2(f"{M1}/{n}", f"{WORK}/{n}")
assert os.path.exists(FSEM_ARCH), FSEM_ARCH
NPY = f"{WORK}/{ESCENA}.npy"

grupos = []
for var in ("original", "adaptado"):
    fssl = f"{DRIVE}/m3/salida/{ESCENA}/{ESCENA}_feats_{var}.npz"
    assert os.path.exists(fssl), f"Falta {fssl}"
    sal = f"{WORK}/m4_{var}"
    print(f"\n=== M4 con f_sem de ArCH + f_ssl {var} ===", flush=True)
    r = subprocess.run([sys.executable, f"{REPO}/experimentacion/modulo4_real/correr_modulo4_real.py",
                        "--npy", NPY, "--fsem", FSEM_ARCH, "--fssl", fssl, "--salida", sal, "--todos",
                        "--k", *map(str, K_LISTA)], cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(r.stdout[-1500:]); assert r.returncode == 0
    grupos.append(f"{var}={sal}/grupos_PL13.npz")
    shutil.copytree(sal, f"{SAL}/m4_{var}", dirs_exist_ok=True)

for nombre_eval, extra in (("evaluacion", []), ("evaluacion_sin_otro", ["--ignorar", "otro"])):
    EV = f"{WORK}/{nombre_eval}"
    r = subprocess.run([sys.executable, f"{REPO}/experimentacion/evaluacion_etiquetada/evaluar_m4.py",
                        "--etiquetas", f"{WORK}/{ESCENA}_etiquetas.npy", "--info", f"{WORK}/{ESCENA}_etiquetas.json",
                        "--grupos", *grupos, "--salida", EV, *extra], cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert r.returncode == 0, r.stdout[-3000:]
    print(open(f"{EV}/evaluacion.md", encoding="utf-8").read())
    shutil.copytree(EV, f"{SAL}/{nombre_eval}", dirs_exist_ok=True)

# zip pequeño para bajar a la laptop
zp = "/content/m4_capilla_arch_resultados.zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for raiz, _, fs in os.walk(SAL):
        for n in fs:
            z.write(os.path.join(raiz, n), os.path.relpath(os.path.join(raiz, n), SAL))
print(round(os.path.getsize(zp) / 2**20, 1), "MB")
from google.colab import files
files.download(zp)
