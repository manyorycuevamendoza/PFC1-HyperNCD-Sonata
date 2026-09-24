#!/usr/bin/env python3
"""Descarga y extrae SemanticPOSS en el layout que espera hyperncd/config/semposs_dataset.yaml.

No requiere login (a diferencia de SemanticKITTI). Uso:

    python scripts/download_semanticposs.py

Corrido desde la raiz de hyperncd/, deja los datos en:

    data/SemanticPOSS/dataset/sequences/00../05/{velodyne,labels}
"""
import argparse
import os
import sys
import urllib.request
import zipfile

URL = "http://www.poss.pku.edu.cn/OpenDataResource/SemanticPOSS/SemanticPOSS_dataset.zip"


def download(url, dest):
    if os.path.exists(dest):
        print(f"{dest} ya existe, salto la descarga.")
        return

    def _progress(block_num, block_size, total_size):
        done = block_num * block_size
        pct = min(100, done * 100 / total_size) if total_size > 0 else 0
        sys.stdout.write(f"\rDescargando {os.path.basename(dest)}: {pct:5.1f}% "
                          f"({done / 1e9:.2f} / {total_size / 1e9:.2f} GB)")
        sys.stdout.flush()

    tmp = dest + ".part"
    urllib.request.urlretrieve(url, tmp, reporthook=_progress)
    print()
    os.rename(tmp, dest)


def extract(zip_path, out_dir):
    print(f"Extrayendo {zip_path} -> {out_dir} ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)
    print("Listo.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data",
                         help="carpeta base donde se guarda el dataset (default: ./data)")
    parser.add_argument("--keep-zip", action="store_true",
                         help="no borrar el .zip tras extraer (~2.4 GB)")
    args = parser.parse_args()

    out_dir = os.path.join(args.data_dir, "SemanticPOSS")
    os.makedirs(out_dir, exist_ok=True)
    zip_path = os.path.join(args.data_dir, "SemanticPOSS_dataset.zip")

    download(URL, zip_path)
    extract(zip_path, out_dir)

    expected = os.path.join(out_dir, "dataset", "sequences", "00", "velodyne")
    if not os.path.isdir(expected):
        print(f"AVISO: no encontre {expected}; revisa el contenido del zip extraido.")
    else:
        print(f"OK: {expected} existe. dataset_path para el yaml: {out_dir}/dataset/")

    if not args.keep_zip:
        os.remove(zip_path)
        print("Zip borrado (usa --keep-zip para conservarlo).")


if __name__ == "__main__":
    main()
