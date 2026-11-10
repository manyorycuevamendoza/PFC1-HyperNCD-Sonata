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

- Checkpoint: `corrida_completa/last.ckpt` (época 30, paso 1230).
- **12.6 %** de los puntos se marcan como novedosos; solo 992,911 como clase conocida.
- **Cabeza dominante: la 3 con el 44.6 % de los puntos** (de 9 cabezas: 6 conocidas y 3 novedosas). Reparto: 0: 122,112, 1: 131,453, 2: 41,936, 3: 506,972, 4: 51,765, 5: 138,673, 6: 56,364, 7: 64,626, 8: 22,045.
- Salud de `f_sem`: rango efectivo 1.7; varianza de 3 y 10 componentes 0.97 y 0.99.
- VRAM pico 3.198 GB, forward 1.677 s, GPU Tesla T4.

**Las cabezas de M2 como clasificador punto a punto** (sin regiones; asignación por mayoría y 1 a 1):

| Evaluación | mIoU (mayoría) | mAcc | allAcc | mIoU 1 a 1 |
|---|---|---|---|---|
| todas las clases | 0.102 | 0.167 | 0.613 | 0.183 |
| sin `otro` | 0.442 | 0.553 | 0.730 | 0.426 |

Porcentaje de cada clase real que M2 marca como novedoso: arco 58.0 %, columna 4.1 %, cupula 17.2 %, pared 1.0 %, techo 3.6 %, otro 13.0 %.

## M3 · Sonata (`f_ssl`)

| Variante | Pesos cargados | Filas de rejilla | VRAM pico (GB) | Forward (s) | Puntos/s |
|---|---|---|---|---|---|
| original | `None` | None | None | None | None |
| adaptado | `None` | None | None | None | None |

Las métricas de M3 con etiquetas están en la evaluación de M4 (módulo `M3`) y en la sonda lineal (`sonda_lineal.py`).

## M4 · Fusión e hipergrafo

| M3 usado | Novedosos según M2 | Regiones | Cobertura de los novedosos | Tiempo (s) |
|---|---|---|---|---|
| original | 12.6 % | 3,831 | 99.5 % | 79 |
| adaptado | 12.6 % | 3,831 | 99.5 % | 68 |
