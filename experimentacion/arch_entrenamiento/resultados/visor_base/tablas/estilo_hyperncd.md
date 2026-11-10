# Tablas al estilo de HyperNCD (IoU %, emparejamiento húngaro 1 a 1)

Como en las Tablas 1 y 2 del paper: IoU por clase tras emparejar grupos con clases, y la columna **Novel** (promedio de las novedosas). Aquí **todas las clases son novedosas** (el modelo no las vio con etiqueta), así que **Known** no existe y **All = Novel**. No hay fila 'Full' (modelo supervisado) porque no se entrenó uno con esta escena.

## Todas las clases · K = 6

| M3 | Método | arco | columna | cupula | pared | techo | otro | Novel | Known | All |
|---|---|---|---|---|---|---|---|---|---|---|
| original | M2 | 22.5 | 2.1 | 46.8 | 54.5 | 23.0 | 35.4 | 30.7 | - | 30.7 |
| original | M3 | 8.4 | 9.9 | 65.6 | 58.8 | 38.2 | 30.7 | 35.3 | - | 35.3 |
| original | M4_base | 22.0 | 0.8 | 29.2 | 40.4 | 33.3 | 23.8 | 24.9 | - | 24.9 |
| original | M4 | 19.3 | 0.3 | 43.4 | 56.5 | 28.7 | 31.3 | 29.9 | - | 29.9 |
| adaptado | M2 | 22.5 | 2.1 | 46.8 | 54.5 | 23.0 | 35.4 | 30.7 | - | 30.7 |
| adaptado | M3 | 8.8 | 7.7 | 16.4 | 54.3 | 0.0 | 27.9 | 19.2 | - | 19.2 |
| adaptado | M4_base | 22.0 | 0.8 | 29.2 | 40.4 | 33.3 | 23.8 | 24.9 | - | 24.9 |
| adaptado | M4 | 16.5 | 4.8 | 38.6 | 57.1 | 20.8 | 25.5 | 27.2 | - | 27.2 |

## Todas las clases · K = 8

| M3 | Método | arco | columna | cupula | pared | techo | otro | Novel | Known | All |
|---|---|---|---|---|---|---|---|---|---|---|
| original | M2 | 23.1 | 6.9 | 39.5 | 55.7 | 30.7 | 20.0 | 29.4 | - | 29.4 |
| original | M3 | 8.0 | 5.4 | 79.4 | 59.2 | 41.8 | 18.9 | 35.5 | - | 35.5 |
| original | M4_base | 24.3 | 4.5 | 40.4 | 52.9 | 35.1 | 23.3 | 30.1 | - | 30.1 |
| original | M4 | 21.6 | 6.0 | 40.2 | 57.8 | 28.6 | 26.9 | 30.2 | - | 30.2 |
| adaptado | M2 | 23.1 | 6.9 | 39.5 | 55.7 | 30.7 | 20.0 | 29.4 | - | 29.4 |
| adaptado | M3 | 12.5 | 7.6 | 18.4 | 54.7 | 22.7 | 19.9 | 22.6 | - | 22.6 |
| adaptado | M4_base | 24.3 | 4.5 | 40.4 | 52.9 | 35.1 | 23.3 | 30.1 | - | 30.1 |
| adaptado | M4 | 16.7 | 5.1 | 38.6 | 56.8 | 38.4 | 24.7 | 30.1 | - | 30.1 |

## Todas las clases · K = 12

| M3 | Método | arco | columna | cupula | pared | techo | otro | Novel | Known | All |
|---|---|---|---|---|---|---|---|---|---|---|
| original | M2 | 15.5 | 7.4 | 39.9 | 55.9 | 35.5 | 19.2 | 28.9 | - | 28.9 |
| original | M3 | 13.0 | 13.8 | 73.7 | 41.1 | 46.4 | 16.6 | 34.1 | - | 34.1 |
| original | M4_base | 22.3 | 6.5 | 46.5 | 53.1 | 43.0 | 13.3 | 30.8 | - | 30.8 |
| original | M4 | 21.6 | 6.1 | 39.5 | 54.4 | 41.4 | 16.8 | 30.0 | - | 30.0 |
| adaptado | M2 | 15.5 | 7.4 | 39.9 | 55.9 | 35.5 | 19.2 | 28.9 | - | 28.9 |
| adaptado | M3 | 11.0 | 10.2 | 41.0 | 54.2 | 24.5 | 15.4 | 26.1 | - | 26.1 |
| adaptado | M4_base | 22.3 | 6.5 | 46.5 | 53.1 | 43.0 | 13.3 | 30.8 | - | 30.8 |
| adaptado | M4 | 23.6 | 9.9 | 39.0 | 56.9 | 27.3 | 17.0 | 29.0 | - | 29.0 |

## Sin la clase `otro` · K = 6

| M3 | Método | arco | columna | cupula | pared | techo | otro | Novel | Known | All |
|---|---|---|---|---|---|---|---|---|---|---|
| original | M2 | 46.3 | 27.2 | 55.1 | 73.2 | 59.6 | 0.0 | 52.3 | - | 52.3 |
| original | M3 | 22.4 | 37.0 | 73.4 | 70.7 | 65.5 | 0.0 | 53.8 | - | 53.8 |
| original | M4_base | 41.6 | 16.7 | 57.6 | 69.3 | 71.4 | 0.0 | 51.3 | - | 51.3 |
| original | M4 | 34.5 | 30.4 | 61.1 | 73.8 | 70.4 | 0.0 | 54.0 | - | 54.0 |
| adaptado | M2 | 46.3 | 27.2 | 55.1 | 73.2 | 59.6 | 0.0 | 52.3 | - | 52.3 |
| adaptado | M3 | 18.4 | 13.8 | 27.1 | 62.0 | 56.1 | 0.0 | 35.5 | - | 35.5 |
| adaptado | M4_base | 41.6 | 16.7 | 57.6 | 69.3 | 71.4 | 0.0 | 51.3 | - | 51.3 |
| adaptado | M4 | 26.8 | 24.9 | 43.9 | 73.0 | 72.6 | 0.0 | 48.2 | - | 48.2 |

## Sin la clase `otro` · K = 8

| M3 | Método | arco | columna | cupula | pared | techo | otro | Novel | Known | All |
|---|---|---|---|---|---|---|---|---|---|---|
| original | M2 | 48.0 | 22.6 | 44.8 | 72.1 | 57.5 | 0.0 | 49.0 | - | 49.0 |
| original | M3 | 18.7 | 28.9 | 80.5 | 68.2 | 66.4 | 0.0 | 52.5 | - | 52.5 |
| original | M4_base | 49.8 | 22.1 | 57.1 | 69.0 | 66.6 | 0.0 | 52.9 | - | 52.9 |
| original | M4 | 42.3 | 28.6 | 43.0 | 74.2 | 65.1 | 0.0 | 50.6 | - | 50.6 |
| adaptado | M2 | 48.0 | 22.6 | 44.8 | 72.1 | 57.5 | 0.0 | 49.0 | - | 49.0 |
| adaptado | M3 | 23.8 | 14.2 | 28.7 | 61.5 | 47.2 | 0.0 | 35.1 | - | 35.1 |
| adaptado | M4_base | 49.8 | 22.1 | 57.1 | 69.0 | 66.6 | 0.0 | 52.9 | - | 52.9 |
| adaptado | M4 | 27.2 | 24.3 | 43.9 | 72.0 | 69.8 | 0.0 | 47.4 | - | 47.4 |

## Sin la clase `otro` · K = 12

| M3 | Método | arco | columna | cupula | pared | techo | otro | Novel | Known | All |
|---|---|---|---|---|---|---|---|---|---|---|
| original | M2 | 27.7 | 22.8 | 42.6 | 71.9 | 51.8 | 0.0 | 43.4 | - | 43.4 |
| original | M3 | 26.3 | 24.3 | 74.2 | 41.8 | 66.6 | 0.0 | 46.6 | - | 46.6 |
| original | M4_base | 31.5 | 17.4 | 51.1 | 67.8 | 62.8 | 0.0 | 46.1 | - | 46.1 |
| original | M4 | 42.5 | 26.7 | 39.9 | 57.6 | 63.6 | 0.0 | 46.1 | - | 46.1 |
| adaptado | M2 | 27.7 | 22.8 | 42.6 | 71.9 | 51.8 | 0.0 | 43.4 | - | 43.4 |
| adaptado | M3 | 20.7 | 18.0 | 46.9 | 59.5 | 39.8 | 0.0 | 37.0 | - | 37.0 |
| adaptado | M4_base | 31.5 | 17.4 | 51.1 | 67.8 | 62.8 | 0.0 | 46.1 | - | 46.1 |
| adaptado | M4 | 27.9 | 22.9 | 41.7 | 71.3 | 60.0 | 0.0 | 44.8 | - | 44.8 |
