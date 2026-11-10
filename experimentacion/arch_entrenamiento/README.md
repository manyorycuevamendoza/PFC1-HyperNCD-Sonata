# Entrenar HyperNCD (M2) con ArCH

**Pregunta.** ¿Un M2 entrenado con patrimonio arquitectónico (ArCH) deja de colapsar en la capilla? Y cuánto tiempo, RAM y GPU cuesta entrenarlo.

| Archivo | Para qué |
|---|---|
| `convertir_arch.py` | `.txt` de ArCH a bloques `.npz` de 10 m: voxel 5 cm, etiqueta por mayoría de votos, reanudable |
| `../../hyperncd/config/arch_dataset.yaml` | clases, `other` ignorada, clases nuevas (`novel_classes`) |
| `../../hyperncd/utils/dataset.py` (`ArCHRestrictedDataset`) | lee los bloques y devuelve lo mismo que el dataset de SemanticPOSS |
| `../../hyperncd/train.py` | acepta `--dataset ArCH` y `--novel` |
| `../monitor_recursos/monitor_recursos.py` | mide tiempo, RAM y GPU de cada comando |
| `../../colab_entrenamiento_arch.ipynb` | convierte, prueba 1 época medida y entrena la corrida completa |

**Decisiones (cambiables)**
- Entrenamiento = carpeta `Training` (15 escenas); validación = `Test` (2). Reparto oficial de ArCH.
- Clases: 9 (`other` ignorada). Nuevas: `arch`, `column`, `vault`. Conocidas: `moldings`, `floor`, `door_window`, `wall`, `stairs`, `roof`.
- Solo geometría (features = unos), como en SemanticPOSS. Se entrena desde cero, igual que la corrida original.
- Muestra = bloque de 10 m, 60 000 puntos al azar por época; las regiones de la clase nueva (DBSCAN) se calculan sobre el bloque completo y se cachean.

**Estado:** el conversor y la lógica del dataset se probaron en local con datos sintéticos (con MinkowskiEngine y torch simulados). Lo que depende de la GPU y de MinkowskiEngine **no está probado**: la primera prueba real es la celda 5 del cuaderno (1 época).

**Resultados:** pendientes.

Licencia de ArCH: CC BY-NC-SA 4.0 (Politecnico di Torino); citar el dataset en la tesis.
