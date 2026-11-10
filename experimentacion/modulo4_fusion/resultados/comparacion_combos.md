# Fusiones de f_sem y f_ssl: ¿alguna gana de forma consistente?

mIoU 1 a 1, K = 8, sin `otro`, media de 5 semillas. Cada celda = **cambio en puntos respecto a la base** (el M4 actual) de esa misma combinación.

Base (M4 actual): HyperNCD original + Sonata original **42.1** · HyperNCD original + Sonata adaptado **33.0** · HyperNCD adaptado + Sonata original **51.7** · HyperNCD adaptado + Sonata adaptado **46.9**

| Configuración | HyperNCD original + Sonata original | HyperNCD original + Sonata adaptado | HyperNCD adaptado + Sonata original | HyperNCD adaptado + Sonata adaptado | media | peor caso | gana en (de 4) | combos con A y B arriba |
|---|---|---|---|---|---|---|---|---|
| `raw|conf` | +6.6 | +5.8 | +1.3 | +1.4 | +3.8 | +1.3 | 4 | 4 |
| `raw|prod|a=0.1` | +6.7 | +4.7 | +1.8 | +1.3 | +3.6 | +1.3 | 4 | 4 |
| `raw|min` | +9.3 | +2.5 | +0.1 | +0.5 | +3.1 | +0.1 | 4 | 2 |
| `raw|prod|a=0.75` | +8.3 | +2.1 | +1.5 | +0.1 | +3.0 | +0.1 | 4 | 3 |
| `center|prod|a=0.5` | +0.0 | +4.9 | +1.4 | +0.1 | +1.6 | +0.0 | 4 | 1 |
| `raw|prod|a=0.9` | +10.2 | +3.7 | +0.7 | -0.5 | +3.5 | -0.5 | 3 | 2 |
| `raw|prod|a=0.5` | +8.5 | +5.1 | -0.0 | -0.9 | +3.2 | -0.9 | 2 | 2 |
| `raw|prod|a=0.25` | +3.5 | +1.2 | +3.0 | -1.3 | +1.6 | -1.3 | 3 | 3 |
| `raw|snf` | +10.6 | +5.2 | +0.5 | -1.7 | +3.6 | -1.7 | 3 | 3 |
| `pca|lineal|a=0.1` | +8.7 | +14.0 | -1.2 | -1.9 | +4.9 | -1.9 | 2 | 2 |
| `center|prod|a=0.75` | -2.4 | +7.1 | +0.9 | -1.1 | +1.1 | -2.4 | 2 | 1 |
| `center|lineal|a=0.5` | -1.9 | +5.3 | -2.2 | -2.6 | -0.3 | -2.6 | 1 | 1 |
| `raw|lineal|a=0.75` | +8.0 | +3.7 | -2.6 | +2.4 | +2.9 | -2.6 | 3 | 3 |
| `center|snf` | -1.3 | +7.3 | -0.5 | -2.8 | +0.7 | -2.8 | 1 | 1 |
| `center|prod|a=0.9` | -3.2 | +6.3 | +1.8 | -3.0 | +0.5 | -3.2 | 2 | 1 |

## Protocolo honesto por combinación (elegir en una mitad, reportar en la otra)

| Combinación | base (media A,B) | estimación honesta | mejora (puntos) | elegida en A | elegida en B |
|---|---|---|---|---|---|
| HyperNCD original + Sonata original | 42.1 | 50.9 | +8.8 | `pca|min` | `raw|snf` |
| HyperNCD original + Sonata adaptado | 33.8 | 52.0 | +18.2 | `pca|prod|a=0.1` | `pca|prod|a=0.1` |
| HyperNCD adaptado + Sonata original | 51.6 | 51.4 | -0.2 | `center|prod|a=0.9|geo=0.25` | `raw|lineal|a=0.1` |
| HyperNCD adaptado + Sonata adaptado | 46.5 | 47.7 | +1.1 | `raw|prod|a=0.1` | `raw|lineal|a=0.75` |

## Medias por familia (cambio respecto a la base, puntos; media de las 4 combinaciones)

| Preprocesado | Modo | media | mejor | peor |
|---|---|---|---|---|
| pca | min | +4.4 | +14.7 | -5.5 |
| raw | conf | +3.8 | +6.6 | +1.3 |
| raw | snf | +3.6 | +10.6 | -1.7 |
| pca | snf | +3.2 | +14.8 | -7.3 |
| raw | min | +3.1 | +9.3 | +0.1 |
| raw | prod | +3.0 | +10.2 | -1.3 |
| pca | prod | +1.2 | +18.9 | -11.5 |
| pca | conf | +0.8 | +13.2 | -8.1 |
| center | snf | +0.7 | +7.3 | -2.8 |
| center | prod | +0.5 | +7.1 | -3.9 |
| raw | lineal | -0.0 | +8.0 | -10.1 |
| pca | lineal | -0.4 | +15.9 | -16.3 |
| center | conf | -1.0 | +4.6 | -4.6 |
| center | min | -1.4 | +5.1 | -4.7 |
| center | lineal | -1.6 | +5.3 | -12.8 |
