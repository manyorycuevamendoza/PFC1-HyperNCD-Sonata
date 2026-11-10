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

**Hecho ya en local (sin GPU), con salidas guardadas**

| Escena | Cabeza dominante | Rango `f_sem` |
|---|---|---|
| Capilla (época 8) | 98,4 % | 3,4 |
| PL13 (época 8) | 82,5 % | 2,1 |
| PL7 (época 8) | 81,4 % | 2,1 |
| PL13 (época 0) | 95,6 % | 8,8 |

Esto muestra que el colapso no es exclusivo de la capilla. **Falta** correr SemanticPOSS y los otros checkpoints en Colab (el control que puede refutar o apoyar la hipótesis del dominio). Hasta entonces, "dominio" es una hipótesis, no un hecho.
