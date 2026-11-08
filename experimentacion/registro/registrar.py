"""Valida los registros de experimentos y regenera el indice que lee el visor.

Cada experimento es un archivo `experimentacion/registro/<id>.json` (ver LEEME.md).
Este script:
  1. comprueba que cada registro tenga los campos obligatorios y que `entradas` apunte a
     experimentos que existen;
  2. avisa de las rutas de resultados que no existen en esta maquina (los .html y datos
     pesados no van a git, asi que en otra laptop es normal que falten);
  3. escribe `indice.json`, `indice.js` (para abrir el visor con doble clic, sin servidor) e
     `INDICE.md` (tabla legible por etapa).

Uso (desde la raiz del repo):
    python experimentacion/registro/registrar.py
"""
import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
ETAPAS = ["M1", "M2", "M3", "M4", "M5"]
OBLIGATORIOS = ["id", "etapa", "titulo", "descripcion", "estado", "fecha", "entradas",
                "parametros", "hardware", "costo", "metricas", "resultados", "datos_pesados", "notas"]


def main():
    registros, errores, avisos = [], [], []
    for f in sorted(AQUI.glob("*.json")):
        if f.name == "indice.json":
            continue
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            errores.append(f"{f.name}: JSON invalido ({e})")
            continue
        faltan = [k for k in OBLIGATORIOS if k not in d]
        if faltan:
            errores.append(f"{f.name}: faltan campos {faltan}")
        if d.get("id") != f.stem:
            errores.append(f"{f.name}: el id ('{d.get('id')}') debe ser igual al nombre del archivo")
        if d.get("etapa") not in ETAPAS:
            errores.append(f"{f.name}: etapa '{d.get('etapa')}' no esta en {ETAPAS}")
        registros.append(d)

    ids = {d.get("id") for d in registros}
    for d in registros:
        for e in d.get("entradas", []):
            if e not in ids:
                errores.append(f"{d['id']}: la entrada '{e}' no es un experimento registrado")
        r = d.get("resultados", {})
        rutas = [r.get("documento")] + r.get("json", []) + r.get("imagenes", []) + r.get("visores", [])
        for p in filter(None, rutas):
            if not (RAIZ / p).exists():
                avisos.append(f"{d['id']}: no existe aqui {p}")

    if errores:
        print("ERRORES:")
        print("\n".join("  - " + e for e in errores))
        sys.exit(1)

    registros.sort(key=lambda d: (ETAPAS.index(d["etapa"]), d["fecha"], d["id"]))
    indice = {"etapas": ETAPAS, "experimentos": registros}
    (AQUI / "indice.json").write_text(json.dumps(indice, indent=1, ensure_ascii=False), encoding="utf-8")
    (AQUI / "indice.js").write_text("const INDICE = " + json.dumps(indice, ensure_ascii=False) + ";\n", encoding="utf-8")

    md = ["# Índice de experimentos", "",
          "Generado por `registrar.py`; no editar a mano. Visor: `experimentacion/visor/index.html`.", ""]
    for et in ETAPAS:
        grupo = [d for d in registros if d["etapa"] == et]
        if not grupo:
            continue
        md += [f"## {et}", "", "| Experimento | Fecha | Estado | Hardware | Entradas |", "|---|---|---|---|---|"]
        for d in grupo:
            md.append(f"| `{d['id']}` — {d['titulo']} | {d['fecha']} | {d['estado']} | {d['hardware']} | "
                      f"{', '.join(d['entradas']) or '-'} |")
        md.append("")
    (AQUI / "INDICE.md").write_text("\n".join(md), encoding="utf-8")

    print(f"{len(registros)} experimentos registrados -> indice.json, indice.js, INDICE.md")
    if avisos:
        print(f"\n{len(avisos)} aviso(s) (normal si faltan .html o datos que no van a git):")
        print("\n".join("  - " + a for a in avisos))


if __name__ == "__main__":
    main()
