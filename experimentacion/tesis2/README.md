# Tesis 2: pendientes, hipótesis y experimentos por hacer

Lista viva de lo que quedó abierto al cierre de Tesis 1 (noviembre de 2026). Cada punto dice **qué
evidencia hay hoy**, **qué falta** y **cómo probarlo**. Lo que no está medido se marca como hipótesis.
Las referencias bibliográficas se citaron de memoria: **verificarlas antes de usarlas en la tesis**.

## 0. Qué está sólido y qué no (al 8 de noviembre de 2026)

| Afirmación | Estado |
|---|---|
| Sonata original da descriptores estables y con estructura en PL13 (M3) | medido |
| El M4 con `f_sem` de la época 8 mezcla de verdad las dos fuentes; con la época 0 era casi una copia de M3 | medido |
| La adaptación de Sonata (lote 1, 21 escaneos, 30 épocas, 3 150 iteraciones) cambia mucho el descriptor (coseno 0,54) y baja su rango efectivo (110 → 81) | medido |
| En M4 la estabilidad baja con el adaptado en 3 de 4 casos | medido, con **3 semillas y un solo entrenamiento** |
| La adaptación "no mejora" o "empeora" el descriptor | **no demostrado**: sin etiquetas y sin control |
| Las bóvedas se sobre-segmentan en M4 y los anillos de la cúpula sí salen coherentes | observación cualitativa, sin número |

Ver `../adaptacion_sonata/README.md` y `../modulo4_real/README.md`.

## 0b. Resultado con etiquetas reales (escena `capilla_etiquetada`, 9-nov-2026)

Ver `../escena_capilla_etiquetada/README.md`. Resumen (sin `otro`, K = 8, 1 a 1, mIoU %): M3 original 53,4 · M3 adaptado 35,6 · M4 original 52,5 · M4 base 32,1 · M2 34,1.
**El Sonata adaptado empeora** (queda como resultado, no se omite); **M2 está colapsado** (98,4 % de los puntos en una cabeza, rango efectivo de `f_sem` 3,4) y por eso **M4 no mejora a M3 solo**.
Ideas que salen de ahí: arreglar M2 (ajuste con etiquetas parciales de la Catedral) antes de seguir con la fusión, y repetir la adaptación con lote mayor.

## 1. Cerrar lo que quedó a medias de la adaptación

1. **Control de entorno.** Extraer PL13 en Colab con los pesos **originales** (`sonata.pth`), con el mismo
   parche SDPA y la misma semilla, y medir el coseno contra el `f_ssl` original de Khipu. Separa el efecto de
   la adaptación del de cambiar de entorno (atención densa en Khipu contra SDPA en Colab; 796 315 contra
   796 309 filas de rejilla).
2. **Variabilidad del entrenamiento.** Repetir la adaptación 2 o 3 veces con semillas distintas. Hoy no se
   sabe si 0,629 contra 0,567 de estabilidad es una diferencia real.
3. **Más semillas en el agrupamiento.** Pasar de 3 a 10 o 20 y reportar media y desviación.
4. **Lote mayor en Khipu** (A6000 de 48 GB: `BATCH=4` o más, `MAX_PUNTOS=65536`). *Hipótesis:* con lote 1 el
   agrupamiento online de Sonata (Sinkhorn sobre el lote) es muy ruidoso y deriva el descriptor. El original
   usa 96 escenas por paso. `sonata_adaptacion/` ya está listo y falta correrlo.
5. **Otras épocas y LR.** Se guardaron `epoch_5` a `epoch_30`: exportar varias y compararlas por mIoU. No hay
   `model_best.pth` porque sin etiquetas no hay métrica.
6. **Completar el costo** (tiempo y VRAM de extracción del adaptado; `train.log`). Ver `datos_costo/`.

## 2. Evaluar con etiquetas

- Terminar de etiquetar el recorte de PL13 (`../etiquetado/`) y calcular **mIoU y matriz de confusión** de M2,
  M3 y M4, original contra adaptado. Es lo que permite decir qué variante acierta más.
- Conseguir más datos etiquetados: asesor, Prof. Daniel y el trabajo de Alvarado y Cuaresma (2024, DGCNN) con
  patio y capilla etiquetados en CloudCompare. Ya se pidieron.
- Con etiquetas, las pruebas dejan de ser solo "comparar variantes entre sí".

## 3. Sobre-segmentación de superficies curvas en M4

*Observación (cualitativa):* los anillos de la cúpula se separan bien, pero las bóvedas se parten en muchos
grupos pequeños (ver `../modulo4_real/README.md`).

1. **Medir el problema sin etiquetas:** componentes conexos por grupo y fracción de regiones vecinas con
   grupos distintos ("fragmentación"). Medirla también por separado en los anillos y las bóvedas.
2. **Continuidad espacial (lo más rápido, sin entrenar):** agrupamiento jerárquico (Ward) restringido a
   regiones vecinas, o suavizado de etiquetas sobre el grafo de vecinos (potencial de Potts con *graph cuts*,
   Boykov et al. 2001; CRF denso, Krähenbühl y Koltun 2011). En HyperNCD encaja como **una hiperarista por
   adyacencia**: posible aporte propio.
3. **Mejores regiones** que las celdas fijas de 0,75 m: crecimiento con restricción de suavidad (Rabbani
   et al. 2006), segmentación por grafo (Felzenszwalb y Huttenlocher 2004), parches localmente convexos
   (LCCP, Stein et al. 2014), supervóxeles (VCCS, Papon et al. 2013).
4. **Sobre-segmentar y fusionar** (superpoints): *Superpoint Graph* (Landrieu y Simonovsky, CVPR 2018) y
   *Superpoint Transformer* (Robert et al., ICCV 2023).
5. **Descriptores menos sensibles a la orientación** (quitar o ponderar menos las normales; tipo de superficie
   en vez de dirección) y **K jerárquico** (fusionar grupos parecidos hasta un umbral).

## 4. ¿Qué tan geométrico o semántico es `f_ssl`?

Sonata se diseñó contra el *geometric shortcut*, pero la adaptación sin etiquetas podría empujarlo de vuelta
hacia geometría o color. Medirlo sin etiquetas: regresión lineal de la altura, la normal vertical, la
planaridad y el color a partir de `f_ssl`, para el original y el adaptado. Si el adaptado se explica más con
geometría, se volvió **más geométrico**, lo contrario de lo buscado.

## 5. Fusión en M4

Variantes baratas por probar: normalizar cada rama, reducir `f_ssl` a 96 dimensiones con PCA, pesos por rama,
fusión tardía de afinidades, confianza de M2 por región y dos hipergrafos. Ver `../modulo4_real/README.md`.

## 6. Un agrupador aprendido (más largo plazo)

Una red pequeña que decida si dos regiones vecinas son la misma estructura. **Necesita supervisión**:
(a) un dataset etiquetado distinto (ScanNet, S3DIS, SemanticPOSS), con el riesgo del cambio de dominio;
(b) el recorte de la Catedral (muy pocos datos); o (c) auto-supervisión. No hay evidencia de que gane a Ward o
Potts; solo tiene sentido si lo sin entrenar se queda corto.

## 7. M2 (HyperNCD)

El modelo se entrenó en calles (SemanticPOSS, mIoU de validación 0,384) y en PL13 el 82 % de los puntos cae en
una sola cabeza novedosa. Ideas: correrlo en los demás escaneos, ajustarlo con etiquetas parciales de la
Catedral, y completar la línea base (mIoU por clase y comparación con el paper).

## 8. Otras ideas discutidas (no evaluadas)

- **Rama de imágenes con DINOv3** como fuente adicional o sustituta de `f_sem`/`f_ssl` (proyectar rasgos 2D a
  3D). Requiere fotos con pose; no se ha probado.
- **Visor liviano** para equipos con poca RAM (puntos en binario, carga bajo demanda, Three.js) y, si hace
  falta, backend y base de datos. Hoy el registro es `../registro/` y el visor `../visor/index.html`.

## 9. Reglas para reportar

1. Reportar resultados negativos o inconclusos (por ejemplo, que la adaptación no mejoró): son resultado.
2. Toda comparación lleva **control, semillas y desviación**, o se dice que no las tiene.
3. Todo experimento nuevo se registra en `../registro/` con su documento y la ruta de sus datos pesados.
4. Declarar el uso de herramientas de IA donde la universidad lo pida.
