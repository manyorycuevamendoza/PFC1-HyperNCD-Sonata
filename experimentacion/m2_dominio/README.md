# Experimento M2: ¿el colapso de HyperNCD es del dominio?

**Pregunta.** En la capilla etiquetada, M2 manda ~98 % de los puntos a una sola cabeza (`f_sem` con rango efectivo 3,4). ¿Es un problema del dominio (SemanticPOSS contra capilla) o del checkpoint, la extracción o el entrenamiento?

**Qué hay aquí**

| Archivo | Para qué |
|---|---|
| `medir_m2.py` | corre M2 (mismo código que `predecir_hyperncd.py`) sobre varias escenas y mide colapso y separación de clases |
| `../../colab_experimento_m2_dominio.ipynb` | cuaderno de Colab que lo ejecuta con 3 checkpoints, muestra los gráficos y baja un zip pequeño |

**Diseño** (ver el cuaderno para el detalle)

1. **Control positivo:** SemanticPOSS secuencia 03 (validación) y 00 (entrenamiento), 8 escaneos cada una. Si ahí también colapsa, el problema **no** es el dominio.
2. **Capilla** y **capilla submuestreada a 80 000 puntos** (densidad de un LiDAR): separa dominio de densidad.
3. **PL7 y PL13** con los `f_sem` ya guardados (época 8).
4. **Tres checkpoints** de `split0_run4`: `best-epoch=8`, `best-epoch=4` y `last`.

**Métricas** (no dependen de qué clase representa cada cabeza, así que son comparables entre dominios): % en la cabeza dominante, cabezas usadas, entropía normalizada, rango efectivo de `f_sem`, y con etiquetas NMI, ARI y mIoU 1 a 1.

## Resultado (corrido en Colab, T4, 9-nov-2026)

Mismo código de inferencia, 8 escaneos por secuencia de SemanticPOSS. mIoU 1 a 1 y NMI entre la cabeza y la clase real.

| Escena | Cabeza dominante | Cabezas > 1 % | Rango `f_sem` | NMI | mIoU 1 a 1 |
|---|---|---|---|---|---|
| SemanticPOSS val (03), época 8 | 30,4 % | 7 | 5,4 | **0,48** | **0,375** |
| SemanticPOSS train (00), época 8 | 28,2 % | 6 | 4,1 | 0,49 | 0,472 |
| Capilla, época 8 | **98,4 %** | 1 | 3,4 | **0,02** | 0,114 |
| Capilla a 80 000 puntos, época 8 | 91,8 % | 4 | 1,6 | 0,03 | 0,099 |
| PL13 / PL7, época 8 | 82,5 % / 81,4 % | 4 | 2,1 | (sin etiquetas) | |

Con la época 4 y con `last` la capilla sigue colapsada (98,2 %; con 80 000 puntos baja a 63,8 % pero NMI 0,08). `last.ckpt` da exactamente lo mismo que la época 4, es decir, **no es la última época**.

**Qué se concluye**
1. **El checkpoint y la extracción de `f_sem` funcionan.** En SemanticPOSS val el mIoU es 0,375, casi el 0,384 guardado en el checkpoint. No es un bug del pipeline ni un checkpoint malo.
2. **El colapso aparece al salir del dominio de entrenamiento**: cabeza dominante 30 % en SemanticPOSS contra 98 % en la capilla y 80 % en la Catedral, y NMI 0,48 contra 0,02.
3. **La densidad de puntos sola no lo explica**: submuestrear la capilla a 80 000 puntos (época 8) sigue dando 92 % en una cabeza.
4. **Corrección:** el rango efectivo bajo no es señal de colapso. En SemanticPOSS también es bajo (4 a 5) y ahí el modelo funciona. Lo que delata el colapso son el reparto de cabezas y el NMI.

## Seguimiento: M2 entrenado en ArCH (10-nov-2026)

Para probar la hipótesis del dominio se entrenó HyperNCD con patrimonio (ArCH) y se midió en la **misma capilla** con este mismo script. Detalle en `../arch_entrenamiento/README.md`.

| M2 | Cabeza dominante | Cabezas > 1 % | NMI | NMI sin «otro» | mIoU 1 a 1 sin «otro» |
|---|---|---|---|---|---|
| SemanticPOSS, época 8 | 98,4 % | 1 | 0,018 | 0,027 | 0,077 |
| **ArCH**, época 30 | **44,6 %** | 9 | 0,191 | **0,439** | **0,355** |

El colapso desaparece al cambiar el dominio de entrenamiento: apoya que la causa era la brecha de dominio y no el pipeline. Límite: una corrida, una semilla, y no se separó qué parte del dominio pesa (patrimonio contra calle, escala, etc.).

**Límites.** "Dominio" agrupa varias cosas que no separamos: escala de la escena, contenido (edificio contra calle) y patrón de un LiDAR real (anillos, caída con la distancia), que el submuestreo al azar no imita. El experimento apoya la hipótesis de la brecha de dominio; no mide cuál de esos factores pesa más. Falta, por ejemplo, reescalar o recortar la capilla, o ajustar M2 con pocas etiquetas de la Catedral.
