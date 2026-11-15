# Sonda lineal de Sonata · `adaptado`

Protocolo del paper de Sonata (clasificador lineal sobre `f_ssl` congelado), con **división espacial** en bloques de 4 m y 2 pliegues (entrena en la mitad A y prueba en la B, y al revés). Valores en %.

## Todas las clases

| Sonda | mIoU | mAcc | allAcc |
|---|---|---|---|
| lineal | 63.7 | 75.6 | 86.0 |
| lineal_balanceada | 64.2 | 77.8 | 85.8 |
| knn | 64.3 | 80.7 | 85.2 |

Por clase (sonda `lineal_balanceada`):

| Clase | Reales | TP | FP | FN | IoU | Precisión | Recall |
|---|---|---|---|---|---|---|---|
| arco | 5,031 | 2,868 | 2,185 | 2,163 | 39.7 | 56.8 | 57.0 |
| columna | 3,442 | 2,014 | 787 | 1,428 | 47.6 | 71.9 | 58.5 |
| cupula | 11,304 | 10,928 | 3,411 | 376 | 74.3 | 76.2 | 96.7 |
| pared | 18,582 | 15,864 | 4,210 | 2,718 | 69.6 | 79.0 | 85.4 |
| techo | 8,069 | 6,568 | 1,013 | 1,501 | 72.3 | 86.6 | 81.4 |
| otro | 73,572 | 64,705 | 5,447 | 8,867 | 81.9 | 92.2 | 87.9 |

## Sin la clase `otro`

| Sonda | mIoU | mAcc | allAcc |
|---|---|---|---|
| lineal | 80.8 | 87.2 | 92.5 |
| lineal_balanceada | 80.5 | 87.2 | 92.2 |
| knn | 82.0 | 89.9 | 92.6 |

Por clase (sonda `lineal_balanceada`):

| Clase | Reales | TP | FP | FN | IoU | Precisión | Recall |
|---|---|---|---|---|---|---|---|
| arco | 13,005 | 10,322 | 1,576 | 2,683 | 70.8 | 86.8 | 79.4 |
| columna | 8,909 | 6,285 | 1,041 | 2,624 | 63.2 | 85.8 | 70.5 |
| cupula | 29,995 | 29,289 | 3,068 | 706 | 88.6 | 90.5 | 97.6 |
| pared | 47,507 | 45,837 | 2,916 | 1,670 | 90.9 | 94.0 | 96.5 |
| techo | 20,584 | 18,950 | 716 | 1,634 | 89.0 | 96.4 | 92.1 |
