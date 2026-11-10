# Tablas al estilo de Sonata (mIoU, mAcc, allAcc, en %)

Mismas tres columnas que las tablas de segmentación semántica del paper de Sonata. **Advertencia:** estos números son de un **agrupamiento sin supervisión** cuyos grupos se asignan a la clase mayoritaria; **no son** el *linear probing* del paper (clasificador lineal entrenado con etiquetas sobre `f_ssl` congelado). Ese protocolo está en `sonda_lineal.py`.

## Todas las clases

| K | M3 | Método | mIoU | mAcc | allAcc | mIoU 1 a 1 |
|---|---|---|---|---|---|---|
| 6 | original | M2 | 28.2 | 38.3 | 71.8 | 30.7 |
| 6 | original | M3 | 32.6 | 40.8 | 75.6 | 35.3 |
| 6 | original | M4_base | 10.2 | 16.7 | 61.2 | 24.9 |
| 6 | original | M4 | 27.7 | 39.6 | 70.6 | 29.9 |
| 6 | adaptado | M2 | 28.2 | 38.3 | 71.8 | 30.7 |
| 6 | adaptado | M3 | 19.9 | 26.5 | 68.6 | 19.2 |
| 6 | adaptado | M4_base | 10.2 | 16.7 | 61.2 | 24.9 |
| 6 | adaptado | M4 | 27.2 | 35.8 | 71.6 | 27.2 |
| 8 | original | M2 | 27.1 | 36.0 | 71.5 | 29.4 |
| 8 | original | M3 | 42.1 | 52.2 | 77.9 | 35.5 |
| 8 | original | M4_base | 26.2 | 36.4 | 69.1 | 30.1 |
| 8 | original | M4 | 30.7 | 40.7 | 74.1 | 30.2 |
| 8 | adaptado | M2 | 27.1 | 36.0 | 71.5 | 29.4 |
| 8 | adaptado | M3 | 20.1 | 26.5 | 68.8 | 22.6 |
| 8 | adaptado | M4_base | 26.2 | 36.4 | 69.1 | 30.1 |
| 8 | adaptado | M4 | 27.1 | 35.6 | 71.6 | 30.1 |
| 12 | original | M2 | 34.3 | 46.0 | 73.0 | 28.9 |
| 12 | original | M3 | 41.4 | 50.7 | 77.9 | 34.1 |
| 12 | original | M4_base | 35.0 | 45.8 | 73.0 | 30.8 |
| 12 | original | M4 | 37.2 | 47.4 | 75.4 | 30.0 |
| 12 | adaptado | M2 | 34.3 | 46.0 | 73.0 | 28.9 |
| 12 | adaptado | M3 | 27.3 | 34.5 | 72.4 | 26.1 |
| 12 | adaptado | M4_base | 35.0 | 45.8 | 73.0 | 30.8 |
| 12 | adaptado | M4 | 35.6 | 45.2 | 75.5 | 29.0 |

Referencia (todo a `otro`): mIoU 10.2, exactitud 61.3.

## Sin la clase `otro`

| K | M3 | Método | mIoU | mAcc | allAcc | mIoU 1 a 1 |
|---|---|---|---|---|---|---|
| 6 | original | M2 | 48.3 | 59.8 | 75.5 | 52.3 |
| 6 | original | M3 | 47.5 | 59.1 | 73.5 | 53.8 |
| 6 | original | M4_base | 48.1 | 60.1 | 74.4 | 51.3 |
| 6 | original | M4 | 48.2 | 60.3 | 75.3 | 54.0 |
| 6 | adaptado | M2 | 48.3 | 59.8 | 75.5 | 52.3 |
| 6 | adaptado | M3 | 35.0 | 49.8 | 59.6 | 35.5 |
| 6 | adaptado | M4_base | 48.1 | 60.1 | 74.4 | 51.3 |
| 6 | adaptado | M4 | 40.7 | 52.8 | 73.1 | 48.2 |
| 8 | original | M2 | 49.1 | 61.0 | 76.6 | 49.0 |
| 8 | original | M3 | 47.3 | 60.1 | 74.8 | 52.5 |
| 8 | original | M4_base | 50.6 | 61.0 | 76.8 | 52.9 |
| 8 | original | M4 | 50.8 | 62.2 | 78.3 | 50.6 |
| 8 | adaptado | M2 | 49.1 | 61.0 | 76.6 | 49.0 |
| 8 | adaptado | M3 | 35.0 | 49.8 | 60.8 | 35.1 |
| 8 | adaptado | M4_base | 50.6 | 61.0 | 76.8 | 52.9 |
| 8 | adaptado | M4 | 40.6 | 52.8 | 73.0 | 47.4 |
| 12 | original | M2 | 47.6 | 59.1 | 75.6 | 43.4 |
| 12 | original | M3 | 51.4 | 62.0 | 76.4 | 46.6 |
| 12 | original | M4_base | 51.0 | 62.6 | 77.7 | 46.1 |
| 12 | original | M4 | 51.9 | 63.4 | 78.8 | 46.1 |
| 12 | adaptado | M2 | 47.6 | 59.1 | 75.6 | 43.4 |
| 12 | adaptado | M3 | 41.0 | 54.1 | 67.9 | 37.0 |
| 12 | adaptado | M4_base | 51.0 | 62.6 | 77.7 | 46.1 |
| 12 | adaptado | M4 | 50.0 | 61.5 | 77.1 | 44.8 |

Referencia (todo a `pared`): mIoU 7.9, exactitud 39.5.
