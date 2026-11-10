# Verificación por etapa — escena `capilla_etiquetada`

Cada bloque dice **qué se comprobó y qué salió**. Los números vienen de las etiquetas reales de la escena.

## M1 · Preprocesamiento

- Puntos: crudo **3,380,661** → tras SOR **3,277,440** (descarta 3.1 %) → tras voxel de 5 cm **1,135,946** (reducción total 66.4 %). Tardó 9.3 s.
- Transferencia de etiquetas: mediana 1.2 cm, p95 2.0 cm, máximo 3.6 cm; puntos sin etiqueta: 0.
- Normales: norma media 1.000; 100.0 % de norma unitaria; 100.0 % orientadas hacia arriba.

| Clase | Puntos crudos | Tras M1 | Conservado | Parte antes | Parte después |
|---|---|---|---|---|---|
| arco | 145,927 | 47,789 | 32.7 % | 4.3 % | 4.2 % |
| columna | 107,105 | 33,278 | 31.1 % | 3.2 % | 2.9 % |
| cupula | 307,330 | 110,152 | 35.8 % | 9.1 % | 9.7 % |
| pared | 527,591 | 173,886 | 33.0 % | 15.6 % | 15.3 % |
| techo | 226,478 | 75,056 | 33.1 % | 6.7 % | 6.6 % |
| otro | 2,066,230 | 695,785 | 33.7 % | 61.1 % | 61.3 % |

Lectura: el voxel reduce más donde hay más densidad; si la **parte** de una clase cambia mucho entre antes y después, M1 la está sesgando.

## M2 · HyperNCD (`f_sem`)

- Checkpoint: `split0_run4/SemanticPOSS_S0_best-epoch=8-step=11196.ckpt` (época 8, paso 11196).
- **99.1 %** de los puntos se marcan como novedosos; solo 9,717 como clase conocida.
- **Cabeza dominante: la 9 con el 98.4 % de los puntos** (de 13 cabezas: 9 conocidas y 4 novedosas). Reparto: 0: 1,750, 1: 935, 2: 345, 3: 268, 4: 6,016, 5: 1, 6: 98, 7: 24, 8: 280, 9: 1,117,335, 10: 267, 11: 6,122, 12: 2,505.
- Salud de `f_sem`: rango efectivo 3.4; varianza de 3 y 10 componentes 0.90 y 0.96.
- VRAM pico 3.198 GB, forward 1.781 s.

**Las cabezas de M2 como clasificador punto a punto** (sin regiones; asignación por mayoría y 1 a 1):

| Evaluación | mIoU (mayoría) | mAcc | allAcc | mIoU 1 a 1 |
|---|---|---|---|---|
| todas las clases | 0.112 | 0.177 | 0.616 | 0.114 |
| sin `otro` | 0.094 | 0.214 | 0.406 | 0.093 |

Porcentaje de cada clase real que M2 marca como novedoso: arco 99.9 %, columna 100.0 %, cupula 99.8 %, pared 98.9 %, techo 99.8 %, otro 99.0 %.

## M3 · Sonata (`f_ssl`)

| Variante | Pesos cargados | Filas de rejilla | VRAM pico (GB) | Forward (s) | Puntos/s |
|---|---|---|---|---|---|
| original | `from HuggingFace: sonata` | 1133238 | 10.49 | 26.6 | 42,644 |
| adaptado | `in local path: /content/drive/MyDrive/PFC1_HyperNCD_Sonata/m3/pesos/sonata_catedral.pth` | 1133238 | 10.49 | 27.0 | 41,959 |

Las métricas de M3 con etiquetas están en la evaluación de M4 (módulo `M3`) y en la sonda lineal (`sonda_lineal.py`).

## M4 · Fusión e hipergrafo

| M3 usado | Novedosos según M2 | Regiones | Cobertura de los novedosos | Tiempo (s) |
|---|---|---|---|---|
| original | 99.1 % | 3,824 | 99.5 % | 80 |
| adaptado | 99.1 % | 3,824 | 99.5 % | 79 |
