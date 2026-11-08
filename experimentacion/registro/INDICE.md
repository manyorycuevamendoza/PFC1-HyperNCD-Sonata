# Índice de experimentos

Generado por `registrar.py`; no editar a mano. Visor: `experimentacion/visor/index.html`.

## M1

| Experimento | Fecha | Estado | Hardware | Entradas |
|---|---|---|---|---|
| `m1_preproceso` — Preprocesamiento de los 23 escaneos (SOR + voxel 5 cm + normales) | 2026-10-08 | hecho | Khipu (CPU, 16 nucleos) | - |

## M2

| Experimento | Fecha | Estado | Hardware | Entradas |
|---|---|---|---|---|
| `m2_epoca0` — HyperNCD f_sem con el checkpoint de la epoca 0 (casi sin entrenar) | 2026-11-06 | descartado como base; se conserva | Colab | - |
| `m2_epoca8` — HyperNCD f_sem con el checkpoint de la epoca 8 (SemanticPOSS split0_run4) | 2026-11-06 | hecho | Colab (MinkowskiEngine) | m1_preproceso |

## M3

| Experimento | Fecha | Estado | Hardware | Entradas |
|---|---|---|---|---|
| `m3_original` — Sonata original (encoder congelado) sobre PL13 | 2026-10-19 | hecho | Khipu RTX A6000 (48 GB) | m1_preproceso |
| `m3_adaptado_e30` — Sonata adaptado a la Catedral (30 epocas, sin etiquetas) | 2026-11-08 | hecho; falta control y mIoU | Colab L4 (22.5 GB) | m1_preproceso |

## M4

| Experimento | Fecha | Estado | Hardware | Entradas |
|---|---|---|---|---|
| `m4_m2e0_m3orig` — Modulo 4 en PL13: f_sem epoca 0 + f_ssl original | 2026-11-06 | hecho (referencia) | Laptop CPU | m2_epoca0, m3_original |
| `m4_m2e8_m3orig` — Modulo 4 en PL13: f_sem epoca 8 + f_ssl original | 2026-11-06 | hecho | Laptop CPU | m2_epoca8, m3_original |
| `m4_m2e8_m3adaptado` — Modulo 4 en PL13: f_sem epoca 8 + f_ssl adaptado | 2026-11-08 | hecho | Laptop CPU | m2_epoca8, m3_adaptado_e30 |
