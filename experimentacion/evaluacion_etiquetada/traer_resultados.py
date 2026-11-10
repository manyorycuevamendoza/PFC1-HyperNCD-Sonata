"""Deja en el repo lo que bajaste del Drive (el paquete que crea `empaquetar.py`).

Destino: experimentacion/escena_<escena>/datos/  (los datos pesados no van a git). Ahi quedan m1/, m2/, m3/, m4/ con la
misma estructura que en el Drive, listos para los cuadros, las comparativas y el visor.

Uso (desde la raiz del repo; por defecto toma el paquete_*.zip mas reciente de tu carpeta Descargas):
    python experimentacion/evaluacion_etiquetada/traer_resultados.py
    python experimentacion/evaluacion_etiquetada/traer_resultados.py --zip C:/ruta/paquete_capilla_etiquetada.zip

Si tienes Google Drive para escritorio (el Drive aparece como una unidad, p. ej. G:), puedes copiar directo, sin zip:
    python experimentacion/evaluacion_etiquetada/traer_resultados.py --drive-desktop "G:/Mi unidad/PFC1_HyperNCD_Sonata"
"""
import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
sys.path.insert(0, str(AQUI))
from empaquetar import listar  # noqa: E402

ESENCIALES = [  # (ruta relativa dentro de la escena, para que sirve)
    ("m1/salida/{e}/{e}.npy", "M1: nube preprocesada"),
    ("m1/salida/{e}/{e}_etiquetas.npy", "M1: etiqueta de cada punto"),
    ("m1/salida/{e}/{e}_etiquetas.json", "M1: clases y control de calidad"),
    ("m2/salida/{e}/{e}_fsem.npz", "M2: f_sem y cabezas"),
    ("m4/salida/{e}/evaluacion/evaluacion.json", "M4: evaluacion con todas las clases"),
    ("m4/salida/{e}/evaluacion_sin_otro/evaluacion.json", "M4: evaluacion sin 'otro'"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", default=None)
    ap.add_argument("--drive-desktop", dest="drive", default=None, help="ruta de PFC1_HyperNCD_Sonata en el Drive de escritorio")
    ap.add_argument("--escena", default="capilla_etiquetada")
    ap.add_argument("--destino", default=None)
    ap.add_argument("--con-features", action="store_true")
    a = ap.parse_args()

    if a.drive:
        escena = a.escena
        destino = Path(a.destino or RAIZ / "experimentacion" / f"escena_{escena}" / "datos")
        pedidos = listar(a.drive, escena, a.con_features)
        if not pedidos:
            raise SystemExit(f"No encontre '{escena}' en {a.drive}")
        for o, d in pedidos:
            (destino / d).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(o, destino / d)
        print(f"Copiados {len(pedidos)} archivos desde {a.drive}")
    else:
        zp = Path(a.zip) if a.zip else None
        if zp is None:
            cand = sorted((Path.home() / "Downloads").glob("paquete_*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not cand:
                raise SystemExit("No hay paquete_*.zip en Descargas. Corre antes la celda de Colab que lo crea, o usa --zip.")
            zp = cand[0]
        with zipfile.ZipFile(zp) as z:
            man = json.loads(z.read("MANIFIESTO.json"))
            escena = man["escena"]
            destino = Path(a.destino or RAIZ / "experimentacion" / f"escena_{escena}" / "datos")
            destino.mkdir(parents=True, exist_ok=True)
            for m in z.namelist():
                if m == "MANIFIESTO.json":
                    continue
                objetivo = (destino / m).resolve()
                assert str(objetivo).startswith(str(destino.resolve())), f"ruta sospechosa en el zip: {m}"
                z.extract(m, destino)
        (destino / "MANIFIESTO.json").write_text(json.dumps(man, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"Paquete: {zp.name}  (creado {man['fecha']})")

    print(f"\nDestino: {destino}")
    print("\nArchivos esenciales:")
    faltan = 0
    for rel, para in ESENCIALES:
        f = destino / rel.format(e=escena)
        ok = f.exists()
        faltan += not ok
        print(("  ✅ " if ok else "  ❌ ") + rel.format(e=escena).split("/", 2)[-1].ljust(46) + (f"{f.stat().st_size / 2**20:8.1f} MB  " if ok else "   FALTA     ") + para)
    m4 = sorted(p.name for p in (destino / "m4" / "salida" / escena).glob("m4_*"))
    print(f"\nM4 por variante de M3: {m4 or 'ninguna'}")
    f3 = sorted(p.name for p in (destino / "m3" / "salida" / escena).glob("*_feats_*.npz"))
    print("f_ssl incluidos:", f3 or "no (son ~2,4 GB cada uno; no hacen falta para tablas y comparativas)")
    print("\n" + ("Todo listo. Siguiente: los cuadros, comparativas y el visor." if not faltan else f"Faltan {faltan} archivos esenciales: revisa que corriste los dos cuadernos completos."))


if __name__ == "__main__":
    main()
