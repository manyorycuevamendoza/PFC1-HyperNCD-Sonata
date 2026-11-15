# Pruebas de los parámetros (P0 a P5)

Pruebas que justifican los valores de `preprocesar.py` (Módulo 1) y `extraer_features.py` (Módulo 3). Cada prueba cambia
**un** parámetro y mide la salida contra una referencia (`--q 20 --lam 2 --voxel 0.05`, normales con 20 vecinos).

**Escena:** la capilla etiquetada («listo», 3 380 661 puntos), la única escena con etiquetas. Las pruebas no usan
etiquetas, solo la geometría.

| Prueba | Qué cambia | Dónde corre | Estado |
|---|---|---|---|
| P0 | tamaño de bloque (1, 5 y 20 millones) | laptop, CPU | **hecha** |
| P1 | filtrado SOR (con o sin, `q`, `lambda`) | laptop, CPU | **hecha** |
| P2 | vóxel (2, 3, 5 y 10 cm) y retención de la rejilla de Sonata | laptop, CPU | **hecha**. Falta el ARI y NMI de los descriptores, que necesita la GPU |
| P3 | vecinos de las normales (`--q-normales`) | laptop, CPU | **hecha** |
| P4 | precisión del `.npz` (media o simple) | GPU | **pendiente**, ver `colab_p4_p5.py` |
| P5 | tamaño de parche (`--parche`) | GPU | **pendiente**, ver `colab_p4_p5.py` |

## Cómo se corre

```bash
python experimentacion/evaluacion_etiquetada/preparar_etiquetas.py unir --carpeta <listo> --salida-las work/capilla.las --salida-etiq work/capilla_etiq.npz
python experimentacion/pruebas_parametros/correr_pruebas.py --las work/capilla.las --trabajo work/trab --salida experimentacion/pruebas_parametros/resultados
```

Deja un `.csv` por prueba y los logs en `resultados/`. Tarda unos 5 minutos en la laptop.

## Medidas

Se calculan sobre una muestra fija de 200 000 puntos de la referencia, emparejada por vecino más cercano.
- `r`, rugosidad: distancia media de cada punto al plano de sus 20 vecinos, en milímetros.
- `disp`, dispersión angular: ángulo entre la normal de cada punto y la normal media de sus 20 vecinos, en grados.
- `ralas`: el decil más disperso de la referencia. `planas`: la mitad con menor curvatura. `aristas`: el decil con mayor curvatura.
- `ret_g`: porcentaje de puntos que conserva una rejilla de lado g (celdas ocupadas sobre puntos).

## Resultados (una escena, una corrida: son medidas, no hay semillas)

**P0.** Con bloques de 1, 5 y 20 millones, `n_sor`, `n_voxel` y la matriz `S` son **idénticos** (`np.array_equal`). El tamaño de bloque solo limita la memoria.

**P1.** El filtrado descarta el 3,05 % de los puntos. En esta escena **no baja la rugosidad** (11,2 mm sin filtrar y 11,9 con el filtrado)
y la dispersión angular cambia poco (17,7° y 16,8°). Su efecto es quitar puntos aislados, no suavizar. Con `q` de 10 a 40 el descarte
va de 3,18 a 2,85 %. Con `lambda` de 1,5 a 3, de 4,72 a 1,49 %.

**P2.** Con vóxel de 2 cm la nube tiene 3 235 662 representantes (2,8 veces la de 5 cm) y la rejilla de Sonata de 2 cm conserva el 98,7 %.
Con 5 cm quedan 1 135 946 y la rejilla de 2 cm conserva el 99,76 %, de modo que **el vóxel es el que fija la resolución**. Con 10 cm la
rugosidad sube a 20,0 mm. Con rejilla de 4 cm sobre el vóxel de 5 cm se conserva el 91,5 %, es decir que se pierde el 8,5 %.

**P3.** La dispersión angular baja al subir los vecinos (de 25,3° con 10 a 11,3° con 40), y en las aristas de 52,3° a 29,5°. **Esa bajada no
es una mejora.** Con más vecinos las normales se suavizan y se parecen más entre sí, por lo que la medida premia el suavizado. Con 10 vecinos
las normales son ruidosas y con 40 se borran las aristas. Se conserva 20 por ser el punto medio y el valor por defecto de Open3D, sin
evidencia de que otro sea mejor.

## Límites
- La medida de dispersión depende de los mismos vecinos que las normales, y no distingue suavizado de acierto. Falta una referencia
  de normales de calidad (por ejemplo, un escaneo más denso) para decidir `q` de las normales.
- Falta P4, P5 y el ARI y NMI de P2. Necesitan la GPU.
- Una sola escena.
