# Tablas al estilo de Sonata (mIoU, mAcc, allAcc, en %)

Mismas tres columnas que las tablas de segmentación semántica del paper de Sonata. **Advertencia:** estos números son de un **agrupamiento sin supervisión** cuyos grupos se asignan a la clase mayoritaria; **no son** el *linear probing* del paper (clasificador lineal entrenado con etiquetas sobre `f_ssl` congelado). Ese protocolo está en `sonda_lineal.py`.

## Todas las clases

| K | M3 | Método | mIoU | mAcc | allAcc | mIoU 1 a 1 |
|---|---|---|---|---|---|---|
| 6 | original | M2 | 18.1 | 26.5 | 65.1 | 20.7 |
| 6 | original | M3 | 30.1 | 39.8 | 74.1 | 32.6 |
| 6 | original | M4_base | 17.9 | 25.8 | 65.4 | 20.2 |
| 6 | original | M4 | 19.0 | 27.0 | 69.9 | 20.5 |
| 6 | adaptado | M2 | 18.1 | 26.5 | 65.1 | 20.7 |
| 6 | adaptado | M3 | 20.1 | 26.7 | 68.6 | 21.7 |
| 6 | adaptado | M4_base | 17.9 | 25.8 | 65.4 | 20.2 |
| 6 | adaptado | M4 | 20.2 | 27.1 | 68.7 | 22.7 |
| 8 | original | M2 | 18.9 | 26.7 | 66.7 | 20.1 |
| 8 | original | M3 | 42.4 | 52.3 | 77.9 | 36.7 |
| 8 | original | M4_base | 17.9 | 25.4 | 65.9 | 20.7 |
| 8 | original | M4 | 42.1 | 52.2 | 77.9 | 36.2 |
| 8 | adaptado | M2 | 18.9 | 26.7 | 66.7 | 20.1 |
| 8 | adaptado | M3 | 20.2 | 26.6 | 68.8 | 22.8 |
| 8 | adaptado | M4_base | 17.9 | 25.4 | 65.9 | 20.7 |
| 8 | adaptado | M4 | 25.4 | 33.7 | 70.8 | 24.5 |
| 12 | original | M2 | 26.3 | 36.5 | 68.1 | 22.2 |
| 12 | original | M3 | 42.7 | 51.1 | 78.8 | 38.0 |
| 12 | original | M4_base | 24.5 | 34.5 | 67.6 | 21.2 |
| 12 | original | M4 | 38.6 | 46.3 | 76.7 | 33.8 |
| 12 | adaptado | M2 | 26.3 | 36.5 | 68.1 | 22.2 |
| 12 | adaptado | M3 | 23.7 | 30.7 | 68.9 | 21.9 |
| 12 | adaptado | M4_base | 24.5 | 34.5 | 67.6 | 21.2 |
| 12 | adaptado | M4 | 24.7 | 31.7 | 70.3 | 24.5 |

Referencia (todo a `otro`): mIoU 10.2, exactitud 61.3.

## Sin la clase `otro`

| K | M3 | Método | mIoU | mAcc | allAcc | mIoU 1 a 1 |
|---|---|---|---|---|---|---|
| 6 | original | M2 | 34.2 | 49.0 | 66.6 | 35.7 |
| 6 | original | M3 | 51.4 | 69.5 | 70.6 | 51.1 |
| 6 | original | M4_base | 28.7 | 43.3 | 59.4 | 33.9 |
| 6 | original | M4 | 26.8 | 41.6 | 53.5 | 29.6 |
| 6 | adaptado | M2 | 34.2 | 49.0 | 66.6 | 35.7 |
| 6 | adaptado | M3 | 29.8 | 45.4 | 60.7 | 35.8 |
| 6 | adaptado | M4_base | 28.7 | 43.3 | 59.4 | 33.9 |
| 6 | adaptado | M4 | 35.4 | 49.7 | 60.0 | 35.3 |
| 8 | original | M2 | 34.2 | 48.6 | 65.1 | 34.1 |
| 8 | original | M3 | 48.2 | 61.4 | 75.5 | 53.4 |
| 8 | original | M4_base | 29.9 | 42.8 | 60.8 | 32.1 |
| 8 | original | M4 | 48.9 | 61.7 | 75.4 | 52.5 |
| 8 | adaptado | M2 | 34.2 | 48.6 | 65.1 | 34.1 |
| 8 | adaptado | M3 | 37.3 | 52.3 | 62.3 | 35.6 |
| 8 | adaptado | M4_base | 29.9 | 42.8 | 60.8 | 32.1 |
| 8 | adaptado | M4 | 33.4 | 49.6 | 61.8 | 36.1 |
| 12 | original | M2 | 33.1 | 46.0 | 61.7 | 34.5 |
| 12 | original | M3 | 48.2 | 58.7 | 75.9 | 50.4 |
| 12 | original | M4_base | 35.2 | 48.4 | 62.1 | 31.8 |
| 12 | original | M4 | 48.3 | 66.1 | 73.2 | 46.4 |
| 12 | adaptado | M2 | 33.1 | 46.0 | 61.7 | 34.5 |
| 12 | adaptado | M3 | 38.0 | 51.8 | 65.2 | 29.3 |
| 12 | adaptado | M4_base | 35.2 | 48.4 | 62.1 | 31.8 |
| 12 | adaptado | M4 | 38.7 | 52.3 | 68.2 | 35.3 |

Referencia (todo a `pared`): mIoU 7.9, exactitud 39.5.
