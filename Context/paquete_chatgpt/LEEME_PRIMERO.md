# Paquete para ChatGPT: Módulo 1 (preprocesamiento) y Módulo 3 (Sonata)

Orden de lectura sugerido:

1. `1_contexto/CONTEXTO_DEFENSA_MODULOS_1_Y_3.md` — todo el procedimiento, las
   cifras, el paralelismo, preguntas probables del profesor y el código.
   Los datos marcados [sin log en el repo] no tienen registro de respaldo.
2. `5_tesis/tesis_borrador.pdf` — la tesis actual (capítulo 3 = marco metodológico).
3. `2_codigo/` — código que se ejecutó:
   - `preprocesar.py` (Módulo 1) y `preprocesar.sbatch` (lanzador en el clúster Khipu).
   - `extraer_features.py` (Módulo 3, Sonata congelado) y `extraer_features.sbatch`.
   - `visualizar_features.py` (PCA, rango efectivo, k-means) y `visor_3d.py` (visor 3D).
   - `sonata/transform.py` y `sonata/model.py`: código de Sonata tal como se usa
     (GridSample, PointTransformerV3, GridPooling). No es código propio.
4. `3_imagenes/` — capturas: `m1_*` son pasos del Módulo 1 sobre PL7; `PL13_*` son
   las etapas de PL13 (crudo, Módulo 1, PCA, similitud, k-means); `lamina_*` son las
   láminas compuestas que van en la tesis.
5. `4_medidas/` — metadatos reales de PL7 y PL13 (conteos tras cada paso) y las
   medidas de los descriptores de Sonata en PL13.

Hechos clave:
- Módulo 1 probado en PL7 (laptop) y ejecutado sobre los 23 LAS en Khipu.
- Sonata ejecutado solo sobre PL13 (803 099 puntos de entrada, 796 315 tras la
  rejilla de 2 cm, salida 796 315 × 1088).
- Sin paralelismo entre archivos ni entre GPU; solo paralelismo interno
  (hilos de Open3D, parches y cabezas de atención en la GPU).
- No hay etiquetas: la calidad de los descriptores aún no se mide con acierto.

## Instrucción sugerida para pegar en ChatGPT

> Soy estudiante de ingeniería y este es el material de mi tesis. Lee primero
> LEEME_PRIMERO.md y el archivo de contexto. Explícame paso a paso el Módulo 1 y
> el Módulo 3 usando el código real del paquete, sin inventar datos que no estén
> en los archivos. Después hazme preguntas como si fueras mi jurado, una por una,
> y corrígeme. Si algo no está en los archivos, dímelo en vez de suponerlo.
