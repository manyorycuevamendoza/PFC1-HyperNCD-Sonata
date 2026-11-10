# Resultados de la escena `capilla_etiquetada`

Todo se regenera con los scripts de `experimentacion/evaluacion_etiquetada/` (comandos en `../README.md`). Valores en % salvo que se indique.

| Carpeta | Contenido |
|---|---|
| `evaluacion/todas/` y `evaluacion/sin_otro/` | `evaluacion.json` (todo, con tablas de cruce), `por_clase.csv` (TP, FP, FN, TN, IoU, precisión, recall, F1, FPR, FNR por clase), `evaluacion.md`, `evaluacion_miou.png` |
| `etapas/` | verificación de M1 y M2 contra las etiquetas: `etapas.md`, `etapas.json`, `m1_clases.png`, `m2_cabezas.png` |
| `tablas/estilo_hyperncd.*` | IoU por clase tras emparejamiento húngaro, con las columnas Novel / Known / All del paper de HyperNCD |
| `tablas/estilo_sonata.*` | mIoU, mAcc y allAcc como en las tablas del paper de Sonata (de un agrupamiento, **no** es linear probing) |
| `tablas/fp_fn_por_clase.*` | falsos positivos y negativos por clase |
| `graficos/` | `miou_por_modulo`, `iou_por_clase`, `fp_fn_por_clase`, `confusion` |

**Evaluaciones:** `todas` cuenta las seis clases; `sin_otro` ignora `otro` (lo no segmentado). **Asignación:** `mayoria` (cada grupo a su clase mayoritaria) y `uno_a_uno` (húngaro).
Una corrida, una escena: sin barras de error.
