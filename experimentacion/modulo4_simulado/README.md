# Módulo 4 simulado con datos controlados

Prueba el **Módulo 4** (fusión, regiones, hipergrafo, prototipos y
pseudo-etiquetas con Sinkhorn) sobre una **escena sintética donde se conoce la
clase de cada punto**, antes de tener los Módulos 2 y 3 reales de la Catedral.

```
python experimentacion/modulo4_simulado/simular_modulo4.py --salida experimentacion/modulo4_simulado/resultados
python experimentacion/modulo4_simulado/graficar.py --dir experimentacion/modulo4_simulado/resultados
```

Tarda unos 3 minutos en la laptop, sin GPU.

## Qué se simula

**Escena.** Nave de 20 × 10 m. Clases **conocidas**: suelo y muros. Clases
**nuevas**: columna, pilar y bóveda (4 tramos). Columna y pilar tienen **la
misma forma** (cilindro de 0,3 m de radio y 6 m de alto) y están intercalados,
así que ni la geometría ni la posición los distinguen. Es el caso que motiva la
tesis: separar cosas que se parecen en forma pero no en significado.

**Descriptores inventados con calidad controlada.** Cada descriptor es el centro
de su clase, más un desvío propio de cada objeto (cada columna concreta se
aparta de su clase), más ruido.

- `f_sem` (96 dim., Módulo 2): separa bien las conocidas y apenas las nuevas,
  porque la red solo vio etiquetas de las conocidas.
- `f_ssl` (1088 dim., Módulo 3): separa las nuevas con una fuerza `sep_ssl`, que
  hace de **calidad de Sonata**.
- `f_geo` (6 dim., línea base): linealidad, planaridad, dispersión y centroide
  de la región, como en la implementación de HyperNCD.

**Módulo 4, con las fórmulas del Capítulo 2.** F = [f_sem ; f_x], regiones con
DBSCAN (radio 0,5 m), afinidad S_b = α S_g + (1−α) S_s, hiperaristas con los M
vecinos más afines, prototipos ponderados por la pertenencia y Sinkhorn
semi-relajado que equilibra **puntos**, no regiones. Se compara la línea base
(f_x = f_geo) con la propuesta (f_x = f_ssl). La métrica es el mIoU de las tres
clases nuevas, con emparejamiento húngaro porque las clases descubiertas no
tienen nombre.

## Lo que NO es

- **No entrena ninguna red.** En HyperNCD la pérdida corrige la red del
  Módulo 2 y las cabezas época a época. Aquí los prototipos se refinan
  alternando asignación y promedio con los descriptores fijos.
- **No predice el mIoU real de la Catedral.** Los descriptores son inventados.
  Lo que muestra es **cómo se comporta** el Módulo 4 y **de qué depende**.

## Resultados (5 repeticiones por condición, `resultados/resultados.csv`)

| Escenario | Qué se varía | Lo que se observa |
|---|---|---|
| Calidad de Sonata | `sep_ssl` 0 a 3 | La línea base queda en 0,35 siempre. La propuesta sube de 0,42 a 0,88: **el aporte de f_ssl depende de que Sonata separe las clases nuevas** |
| Peso de la geometría α | 0 a 1 | Con α = 1 (solo geometría) las dos versiones dan 0,51, control de cordura. La propuesta mejora al bajar α: lo que aporta está en S_s, no en S_g |
| Vecinos por hiperarista M | 1 a 16 | La propuesta mejora con M (0,51 → 0,83). Con solo 12 regiones, M = 16 equivale a usar todas |
| Desbalance | puntos de pilar, 100 % a 10 % | La propuesta cae de 0,75 a 0,52: Sinkhorn reparte por igual y **parte la clase grande** cuando una clase es pequeña |
| Ruido en coordenadas | 0,5 a 10 cm | Sin efecto: el radio de DBSCAN (0,5 m) es mucho mayor |

**Variabilidad alta.** Las bandas de `curvas.png` son la desviación entre
repeticiones y son anchas: con solo 12 regiones, una región mal asignada mueve
mucho el mIoU. `escena.png` muestra **una** repetición (la propuesta acertó
todo); el promedio en esa condición es 0,35 frente a 0,75, y así lo indica el
título de la figura.

## Caso concreto: qué miden S_g y S_s

- **S_g (geométrica)** responde a "¿se parecen en forma y en posición?". Usa solo
  g: linealidad, planaridad, dispersión y centroide de la región.
- **S_s (semántica)** responde a "¿se parecen en lo que son?". Usa el descriptor
  F = [f_sem ; f_x], y es por donde entra Sonata.

Repetición 1, calidad de Sonata 1,5 (`caso_ss_sg.py` lo recalcula). Primero la
forma de tres regiones:

| Región | Linealidad | Planaridad | Dispersión |
|---|---|---|---|
| Columna | 0,98 | 0,00 | 0,01 |
| Pilar | 0,98 | 0,00 | 0,01 |
| Bóveda | 0,90 | 0,04 | 0,07 |

Columna y pilar tienen la misma forma. Ahora las afinidades:

| Par | S_g | S_s línea base (f_geo) | S_s propuesta (f_ssl) |
|---|---|---|---|
| Columna vs pilar | 0,37 | 0,79 | 0,54 |
| Columna vs bóveda | 0,05 | 0,94 | 0,55 |

- **S_g** separa bien columna y bóveda (0,05). Entre columna y pilar da 0,37 y no
  1, aunque la forma sea idéntica, porque g incluye el centroide y los dos están
  en lugares distintos. S_g mezcla forma y posición.
- **S_s de la línea base** dice que la columna se parece más a la bóveda (0,94)
  que al pilar (0,79). Por eso la línea base confunde parte de la bóveda con
  "pilar": f_geo no sabe qué es cada cosa y f_sem no distingue clases nuevas.
- **S_s de la propuesta** quita esas similitudes falsas (0,79 → 0,54 y
  0,94 → 0,55). En este par concreto no deja la columna más cerca del pilar que
  de la bóveda: la mejora aparece al combinar **todas** las regiones en el
  hipergrafo, por eso se lee el promedio (0,35 → 0,75) y no un par suelto.

En una frase: S_g mide si dos regiones se parecen por fuera, S_s si se parecen
por dentro. Cuando la forma es igual, solo S_s puede separarlas, y solo si recibe
un buen descriptor.

## Archivos

- `simular_modulo4.py`: escena, descriptores, Módulo 4, métricas y escenarios.
- `graficar.py`: `curvas.png`, `escena.png` y `escena_3d.html` (3D interactivo,
  doble clic para abrirlo).
- `figuras_pasos.py`: una figura por paso para la tesis, `paso2_regiones.png`,
  `paso3_afinidad.png` y `paso4_hiperarista.png`. Las copias de la tesis están en
  `LATEX/asesoria5/imagenes/m4_*.png`, recortadas de blanco.
- `caso_ss_sg.py`: recalcula la tabla del caso concreto de S_g y S_s.

La tesis lo documenta paso a paso en el Capítulo 3, sección "Prueba Simulada
del Módulo 4", con la misma estructura que los Módulos 1 y 3.
- `resultados/`: lo generado.
