"""Empaqueta, desde el Drive, SOLO lo necesario de una escena para hacer cuadros, comparativas y el visor en la laptop.

Incluye (por defecto, ~200-300 MB):
    m1/salida/<escena>/        M1: .npy, .json y etiquetas por punto
    m1/entrada/<escena>/       etiquetas crudas (.npz); el .las se deja fuera
    m2/salida/<escena>/        M2: f_sem (.npz), .json y .csv
    m3/salida/<escena>/        solo los .txt/.json pequenos (con que pesos se calculo cada f_ssl)
    m4/salida/<escena>/        M4 de cada variante, evaluaciones (con FP/FN), tablas y logs
NO incluye los f_ssl (.npz de ~2,4 GB cada uno) salvo con --con-features.

Uso en Colab o en una maquina con el Drive montado:
    python empaquetar.py --drive /content/drive/MyDrive/PFC1_HyperNCD_Sonata --escena capilla_etiquetada
"""
import argparse
import json
import time
import zipfile
from pathlib import Path


def listar(drive, escena, con_features=False):
    """Devuelve [(ruta_origen, ruta_dentro_del_zip)]."""
    D = Path(drive)
    out = []

    def agregar(carpeta, filtro=None):
        base = D / carpeta
        if not base.exists():
            return
        for f in sorted(base.rglob("*")):
            if f.is_file() and (filtro is None or filtro(f)):
                out.append((f, str(f.relative_to(D)).replace("\\", "/")))

    agregar(f"m1/salida/{escena}")
    agregar(f"m1/entrada/{escena}", lambda f: f.suffix == ".npz")          # etiquetas crudas; el .las pesa mucho y no hace falta
    agregar(f"m2/salida/{escena}")
    agregar(f"m3/salida/{escena}", (lambda f: True) if con_features else (lambda f: f.suffix in (".txt", ".json", ".csv")))
    agregar(f"m4/salida/{escena}")
    return out


def empaquetar(drive, escena, salida, con_features=False):
    pedidos = listar(drive, escena, con_features)
    if not pedidos:
        raise SystemExit(f"No encontre nada de '{escena}' en {drive}. ¿Corriste los cuadernos 1 y 2?")
    salida = Path(salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    total = sum(o.stat().st_size for o, _ in pedidos)
    manifiesto = {"escena": escena, "fecha": time.strftime("%Y-%m-%d %H:%M"), "con_features": con_features,
                  "archivos": [{"ruta": d, "MB": round(o.stat().st_size / 2 ** 20, 2)} for o, d in pedidos]}
    with zipfile.ZipFile(salida, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as z:
        z.writestr("MANIFIESTO.json", json.dumps(manifiesto, indent=1, ensure_ascii=False))
        for i, (o, d) in enumerate(pedidos, 1):
            z.write(o, d)
            print(f"  [{i}/{len(pedidos)}] {d}  ({o.stat().st_size / 2 ** 20:.1f} MB)")
    print(f"\n✅ {salida}  ({salida.stat().st_size / 2 ** 20:.0f} MB comprimido; {total / 2 ** 20:.0f} MB sin comprimir)")
    return salida


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drive", required=True, help="carpeta PFC1_HyperNCD_Sonata dentro del Drive")
    ap.add_argument("--escena", default="capilla_etiquetada")
    ap.add_argument("--salida", default=None, help="por defecto <drive>/paquetes/paquete_<escena>.zip")
    ap.add_argument("--con-features", action="store_true", help="incluir los f_ssl (~2,4 GB cada uno)")
    a = ap.parse_args()
    empaquetar(a.drive, a.escena, a.salida or f"{a.drive}/paquetes/paquete_{a.escena}.zip", a.con_features)
