# Índice de experimentos

Generado por `registrar.py`; no editar a mano. Visor: `experimentacion/visor/index.html`.

## M1

| Experimento | Fecha | Estado | Hardware | Entradas |
|---|---|---|---|---|
| `m1_preproceso` — Preprocesamiento de los 23 escaneos (SOR + voxel 5 cm + normales) | 2026-10-08 | hecho | Khipu (CPU, 16 nucleos) | - |
| `escena_m1` — Escena etiquetada `listo`: preprocesamiento | 2026-11-09 | hecho (1 corrida, 1 escena) | Colab (CPU) | - |

## M2

| Experimento | Fecha | Estado | Hardware | Entradas |
|---|---|---|---|---|
| `m2_epoca0` — HyperNCD f_sem con el checkpoint de la epoca 0 (casi sin entrenar) | 2026-11-06 | descartado como base; se conserva | Colab | - |
| `m2_epoca8` — HyperNCD f_sem con el checkpoint de la epoca 8 (SemanticPOSS split0_run4) | 2026-11-06 | hecho | Colab (MinkowskiEngine) | m1_preproceso |
| `escena_m2` — Escena etiquetada: HyperNCD (f_sem), epoca 8 | 2026-11-09 | hecho (1 corrida, 1 escena) | Colab | escena_m1 |

## M3

| Experimento | Fecha | Estado | Hardware | Entradas |
|---|---|---|---|---|
| `m3_original` — Sonata original (encoder congelado) sobre PL13 | 2026-10-19 | hecho | Khipu RTX A6000 (48 GB) | m1_preproceso |
| `m3_adaptado_e30` — Sonata adaptado a la Catedral (30 epocas, sin etiquetas) | 2026-11-08 | hecho; falta control y mIoU | Colab L4 (22.5 GB) | m1_preproceso |
| `escena_m3_adaptado` — Escena etiquetada: Sonata adaptado a la Catedral | 2026-11-09 | hecho (1 corrida, 1 escena) | Colab L4 | escena_m1 |
| `escena_m3_original` — Escena etiquetada: Sonata original | 2026-11-09 | hecho (1 corrida, 1 escena) | Colab L4 | escena_m1 |

## M4

| Experimento | Fecha | Estado | Hardware | Entradas |
|---|---|---|---|---|
| `m4_m2e0_m3orig` — Modulo 4 en PL13: f_sem epoca 0 + f_ssl original | 2026-11-06 | hecho (referencia) | Laptop CPU | m2_epoca0, m3_original |
| `m4_m2e8_m3orig` — Modulo 4 en PL13: f_sem epoca 8 + f_ssl original | 2026-11-06 | hecho | Laptop CPU | m2_epoca8, m3_original |
| `m4_m2e8_m3adaptado` — Modulo 4 en PL13: f_sem epoca 8 + f_ssl adaptado | 2026-11-08 | hecho | Laptop CPU | m2_epoca8, m3_adaptado_e30 |
| `escena_m4_m3adaptado` — Escena etiquetada: M4 con Sonata adaptado | 2026-11-09 | hecho (1 corrida, 1 escena) | Colab (CPU) | escena_m2, escena_m3_adaptado |
| `escena_m4_m3original` — Escena etiquetada: M4 con Sonata original | 2026-11-09 | hecho (1 corrida, 1 escena) | Colab (CPU) | escena_m2, escena_m3_original |
