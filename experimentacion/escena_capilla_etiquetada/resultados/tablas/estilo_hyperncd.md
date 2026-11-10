# Tablas al estilo de HyperNCD (IoU %, emparejamiento húngaro 1 a 1)

Como en las Tablas 1 y 2 del paper: IoU por clase tras emparejar grupos con clases, y la columna **Novel** (promedio de las novedosas). Aquí **todas las clases son novedosas** (el modelo no las vio con etiqueta), así que **Known** no existe y **All = Novel**. No hay fila 'Full' (modelo supervisado) porque no se entrenó uno con esta escena.

## Todas las clases · K = 6

| M3 | Método | arco | columna | cupula | pared | techo | otro | Novel | Known | All |
|---|---|---|---|---|---|---|---|---|---|---|
| original | M2 | 4.3 | 2.9 | 19.3 | 46.8 | 24.6 | 26.1 | 20.7 | - | 20.7 |
| original | M3 | 8.1 | 9.9 | 50.0 | 59.9 | 37.7 | 30.2 | 32.6 | - | 32.6 |
| original | M4_base | 5.9 | 3.9 | 23.4 | 44.9 | 22.1 | 20.7 | 20.2 | - | 20.2 |
| original | M4 | 7.1 | 3.9 | 5.8 | 41.9 | 36.9 | 27.6 | 20.5 | - | 20.5 |
| adaptado | M2 | 4.3 | 2.9 | 19.3 | 46.8 | 24.6 | 26.1 | 20.7 | - | 20.7 |
| adaptado | M3 | 6.5 | 3.5 | 20.9 | 55.4 | 21.4 | 22.6 | 21.7 | - | 21.7 |
| adaptado | M4_base | 5.9 | 3.9 | 23.4 | 44.9 | 22.1 | 20.7 | 20.2 | - | 20.2 |
| adaptado | M4 | 6.5 | 4.4 | 21.4 | 56.2 | 22.4 | 25.2 | 22.7 | - | 22.7 |

## Todas las clases · K = 8

| M3 | Método | arco | columna | cupula | pared | techo | otro | Novel | Known | All |
|---|---|---|---|---|---|---|---|---|---|---|
| original | M2 | 5.6 | 4.8 | 12.2 | 50.0 | 28.6 | 19.2 | 20.1 | - | 20.1 |
| original | M3 | 10.1 | 10.1 | 80.3 | 61.0 | 40.8 | 17.7 | 36.7 | - | 36.7 |
| original | M4_base | 6.6 | 3.4 | 21.6 | 44.5 | 30.3 | 17.5 | 20.7 | - | 20.7 |
| original | M4 | 8.3 | 9.6 | 79.2 | 60.3 | 40.8 | 18.9 | 36.2 | - | 36.2 |
| adaptado | M2 | 5.6 | 4.8 | 12.2 | 50.0 | 28.6 | 19.2 | 20.1 | - | 20.1 |
| adaptado | M3 | 13.0 | 7.7 | 19.2 | 55.4 | 21.9 | 19.4 | 22.8 | - | 22.8 |
| adaptado | M4_base | 6.6 | 3.4 | 21.6 | 44.5 | 30.3 | 17.5 | 20.7 | - | 20.7 |
| adaptado | M4 | 10.2 | 9.4 | 34.9 | 48.3 | 22.5 | 21.9 | 24.5 | - | 24.5 |

## Todas las clases · K = 12

| M3 | Método | arco | columna | cupula | pared | techo | otro | Novel | Known | All |
|---|---|---|---|---|---|---|---|---|---|---|
| original | M2 | 0.7 | 5.2 | 10.7 | 52.2 | 37.7 | 26.4 | 22.2 | - | 22.2 |
| original | M3 | 13.1 | 14.5 | 80.9 | 56.6 | 45.3 | 17.5 | 38.0 | - | 38.0 |
| original | M4_base | 6.7 | 5.0 | 21.5 | 45.8 | 37.4 | 10.7 | 21.2 | - | 21.2 |
| original | M4 | 11.3 | 16.4 | 63.8 | 51.6 | 44.4 | 15.6 | 33.8 | - | 33.8 |
| adaptado | M2 | 0.7 | 5.2 | 10.7 | 52.2 | 37.7 | 26.4 | 22.2 | - | 22.2 |
| adaptado | M3 | 12.0 | 7.8 | 25.4 | 51.1 | 20.1 | 15.2 | 21.9 | - | 21.9 |
| adaptado | M4_base | 6.7 | 5.0 | 21.5 | 45.8 | 37.4 | 10.7 | 21.2 | - | 21.2 |
| adaptado | M4 | 11.6 | 13.5 | 26.2 | 55.1 | 25.1 | 15.4 | 24.5 | - | 24.5 |

## Sin la clase `otro` · K = 6

| M3 | Método | arco | columna | cupula | pared | techo | Novel | Known | All |
|---|---|---|---|---|---|---|---|---|---|
| original | M2 | 9.5 | 15.1 | 35.1 | 66.1 | 52.8 | 35.7 | - | 35.7 |
| original | M3 | 21.4 | 39.7 | 62.0 | 68.6 | 64.1 | 51.1 | - | 51.1 |
| original | M4_base | 13.8 | 10.0 | 39.6 | 60.0 | 46.0 | 33.9 | - | 33.9 |
| original | M4 | 14.2 | 17.0 | 11.0 | 43.6 | 62.4 | 29.6 | - | 29.6 |
| adaptado | M2 | 9.5 | 15.1 | 35.1 | 66.1 | 52.8 | 35.7 | - | 35.7 |
| adaptado | M3 | 20.2 | 17.0 | 34.5 | 63.2 | 43.8 | 35.8 | - | 35.8 |
| adaptado | M4_base | 13.8 | 10.0 | 39.6 | 60.0 | 46.0 | 33.9 | - | 33.9 |
| adaptado | M4 | 20.9 | 9.7 | 35.7 | 65.5 | 44.9 | 35.3 | - | 35.3 |

## Sin la clase `otro` · K = 8

| M3 | Método | arco | columna | cupula | pared | techo | Novel | Known | All |
|---|---|---|---|---|---|---|---|---|---|
| original | M2 | 12.6 | 11.1 | 23.3 | 64.9 | 58.5 | 34.1 | - | 34.1 |
| original | M3 | 18.4 | 32.1 | 81.0 | 69.1 | 66.3 | 53.4 | - | 53.4 |
| original | M4_base | 13.9 | 8.7 | 30.0 | 56.1 | 51.7 | 32.1 | - | 32.1 |
| original | M4 | 18.8 | 28.2 | 80.5 | 69.6 | 65.3 | 52.5 | - | 52.5 |
| adaptado | M2 | 12.6 | 11.1 | 23.3 | 64.9 | 58.5 | 34.1 | - | 34.1 |
| adaptado | M3 | 25.3 | 14.3 | 29.3 | 61.9 | 47.3 | 35.6 | - | 35.6 |
| adaptado | M4_base | 13.9 | 8.7 | 30.0 | 56.1 | 51.7 | 32.1 | - | 32.1 |
| adaptado | M4 | 19.6 | 23.8 | 41.0 | 52.8 | 43.2 | 36.1 | - | 36.1 |

## Sin la clase `otro` · K = 12

| M3 | Método | arco | columna | cupula | pared | techo | Novel | Known | All |
|---|---|---|---|---|---|---|---|---|---|
| original | M2 | 12.2 | 11.3 | 27.9 | 62.1 | 58.7 | 34.5 | - | 34.5 |
| original | M3 | 20.5 | 24.3 | 81.2 | 59.4 | 66.6 | 50.4 | - | 50.4 |
| original | M4_base | 13.1 | 10.2 | 28.0 | 52.3 | 55.4 | 31.8 | - | 31.8 |
| original | M4 | 21.0 | 32.3 | 64.3 | 53.0 | 61.1 | 46.4 | - | 46.4 |
| adaptado | M2 | 12.2 | 11.3 | 27.9 | 62.1 | 58.7 | 34.5 | - | 34.5 |
| adaptado | M3 | 20.9 | 11.6 | 32.2 | 55.0 | 26.8 | 29.3 | - | 29.3 |
| adaptado | M4_base | 13.1 | 10.2 | 28.0 | 52.3 | 55.4 | 31.8 | - | 31.8 |
| adaptado | M4 | 20.9 | 26.5 | 30.8 | 60.6 | 37.8 | 35.3 | - | 35.3 |
