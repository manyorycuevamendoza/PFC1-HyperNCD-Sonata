# Sonda lineal de Sonata · `original`

Protocolo del paper de Sonata (clasificador lineal sobre `f_ssl` congelado), con **división espacial** en bloques de 4 m y 2 pliegues (entrena en la mitad A y prueba en la B, y al revés). Valores en %.

## Todas las clases

| Sonda | mIoU | mAcc | allAcc |
|---|---|---|---|
| lineal | 59.3 | 72.9 | 83.2 |
| lineal_balanceada | 59.6 | 76.0 | 83.0 |
| knn | 65.4 | 80.8 | 86.1 |

Por clase (sonda `lineal_balanceada`):

| Clase | Reales | TP | FP | FN | IoU | Precisión | Recall |
|---|---|---|---|---|---|---|---|
| arco | 5,031 | 2,742 | 2,677 | 2,289 | 35.6 | 50.6 | 54.5 |
| columna | 3,442 | 2,026 | 2,086 | 1,416 | 36.6 | 49.3 | 58.9 |
| cupula | 11,304 | 10,705 | 3,273 | 599 | 73.4 | 76.6 | 94.7 |
| pared | 18,582 | 14,981 | 4,724 | 3,601 | 64.3 | 76.0 | 80.6 |
| techo | 8,069 | 6,660 | 1,586 | 1,409 | 69.0 | 80.8 | 82.5 |
| otro | 73,572 | 62,527 | 6,013 | 11,045 | 78.6 | 91.2 | 85.0 |

## Sin la clase `otro`

| Sonda | mIoU | mAcc | allAcc |
|---|---|---|---|
| lineal | 79.0 | 87.0 | 91.4 |
| lineal_balanceada | 79.7 | 87.6 | 91.9 |
| knn | 81.2 | 89.9 | 92.3 |

Por clase (sonda `lineal_balanceada`):

| Clase | Reales | TP | FP | FN | IoU | Precisión | Recall |
|---|---|---|---|---|---|---|---|
| arco | 13,005 | 9,734 | 2,543 | 3,271 | 62.6 | 79.3 | 74.8 |
| columna | 8,909 | 6,896 | 1,474 | 2,013 | 66.4 | 82.4 | 77.4 |
| cupula | 29,995 | 28,774 | 2,152 | 1,221 | 89.5 | 93.0 | 95.9 |
| pared | 47,507 | 45,398 | 2,285 | 2,109 | 91.2 | 95.2 | 95.6 |
| techo | 20,584 | 19,435 | 1,309 | 1,149 | 88.8 | 93.7 | 94.4 |
