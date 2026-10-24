# Guía del proyecto PFC1 — HyperNCD + Sonata

Revisión del repositorio local realizada el 16 de octubre de 2026. Este archivo está pensado para entregarlo a otro chat y pedirle una explicación progresiva. El directorio revisado se llama `PFC1-HyperNCD-Sonata`; no se encontró una carpeta independiente llamada `PFC1-HyperNCD-Sonata_1`.

## 1. Cómo interpretar esta revisión

Se leyó código, documentación LaTeX y archivos pequeños de contexto. **No se modificó código, no se instalaron dependencias ni se ejecutaron el preprocesamiento o los modelos.** Para contrastar los datos grandes solo se leyeron sus cabeceras y 100 registros puntuales del LAS y del NPY mediante accesos directos; no se cargaron las nubes completas ni se abrió el ZIP.

Las referencias `ruta:línea` corresponden al código local en esta revisión. Un intervalo escrito en el texto identifica el fragmento relevante; las líneas pueden cambiar en futuras versiones.

Se distinguen cuatro tipos de evidencia:

- **Código:** comportamiento visible al leer la implementación; no demuestra que una corrida haya terminado correctamente.
- **Comprobado en esta revisión:** cabeceras, tamaños, muestras y cálculos pequeños que sí se contrastaron ahora.
- **Documentado:** cifras o afirmaciones guardadas en el proyecto, que no se reprodujeron ahora.
- **Propuesto / pendiente:** diseño sin conexión completa o sin resultados verificables encontrados.

## 2. Objetivo de la tesis

El título actual es «Modelo híbrido HyperNCD-Sonata para el descubrimiento de clases novedosas en la segmentación semántica de nubes de puntos de patrimonio arquitectónico» (`LATEX/asesoria5/main.tex:68–70`).

La tesis busca construir y medir un modelo que combine HyperNCD con representaciones aprendidas por Sonata, para clasificar elementos estructurales y descubrir clases novedosas en el levantamiento de la Catedral de Lima. La mejora se debe cuantificar mediante mIoU, especialmente en las clases novedosas (`LATEX/asesoria5/secciones/introduccion.tex:69–89`).

En términos sencillos:

1. El escáner entrega millones de puntos con posición y color.
2. Se limpian y reducen esos puntos.
3. HyperNCD produce un descriptor semántico: información aprendida para distinguir clases.
4. Sonata produce otro descriptor, aprendido de forma autosupervisada y usado sin entrenar sus pesos en esta extracción.
5. Se propone sustituir el descriptor geométrico manual de HyperNCD por el de Sonata, conservando el razonamiento por hipergrafo.
6. Se compara el híbrido con HyperNCD sin esa sustitución.

La fórmula de diseño es `F_i = [f_sem(x_i); f_ssl(x_i)]`, frente a la base `[f_sem; f_geo]`. Un descriptor es un vector de números que resume características; no es todavía una etiqueta como «columna» o «arco».

En la introducción actual, **OE1** es integrar Sonata, **OE2** calibrar la configuración y **OE3** analizar la mejora (`introduccion.tex:94–140`). El `README.md:3–6` todavía llama **OE2** a la integración: hay una diferencia de numeración entre documentos. Conviene explicar el contenido del objetivo antes que apoyarse solo en su número.

## 3. Estructura y archivos principales

```text
PFC1-HyperNCD-Sonata/
├── README.md                         Idea de integración y dos entornos
├── SETUP_KHIPU.md                    Instrucciones del clúster
├── setup.sh                         Preparación de entornos/repositorios
├── preprocesar.py                   Módulo 1 propio
├── preprocesar.sbatch               Preprocesamiento por archivos en Slurm
├── extraer_features.py              Extracción y caché de Sonata
├── extraer_features.sbatch          Lanzador GPU de esa extracción
├── demo_sonata.sbatch               Demo de segmentación de Sonata
├── hyperncd_train_poss.sbatch        Entrenamiento base en SemanticPOSS
├── hyperncd_install_mke.sbatch       Apoyo para instalar MinkowskiEngine
├── Context/
│   ├── inspeccionar_data.py         Inspección previa de los datos
│   ├── inspeccion_DATA.json         Reporte previo de esa inspección
│   ├── DATA.zip                     Copia comprimida grande; no necesaria
│   └── DATA/                        PL7, salidas, metadatos y muestras
├── hyperncd/
│   ├── train.py                     Entrada de entrenamiento con Discoverer_hyper
│   ├── train_KITTI.py, train_GAP.py  Otras entradas de entrenamiento
│   ├── test.py, test_GAP.py          Entradas de evaluación
│   ├── models/                      MinkUNet y cabezas de clasificación
│   ├── modules/                     Entrenamiento, regiones e hipergrafo
│   ├── utils/                       Lectores, cuantización, métricas y Sinkhorn
│   ├── config/                      SemanticKITTI y SemanticPOSS
│   ├── scripts/                     Descarga de SemanticPOSS
│   ├── logs/                        Registros existentes de experimentos base
│   └── region_save/                 Cachés de regiones .pkl
├── sonata/
│   ├── sonata/                      PTv3, transformaciones y estructuras
│   ├── demo/                        PCA, similitud, segmentación e inferencia
│   ├── README.md, setup.py, environment.yml
│   └── assets/                      Recursos del repositorio
├── LATEX/
│   ├── asesoria1.1.tex, asesoria1.2.tex, asesoria2.tex, asesoria3.tex
│   └── asesoria5/
│       ├── main.tex                 Fuente principal actual
│       ├── secciones/               Objetivos, teoría, método y anexos
│       ├── imagenes/                Diagramas y figura de PL7
│       └── referencias.bib          Bibliografía
├── logs/                            Dos pruebas de demo de Sonata
└── paper de referencia/             Artículos PDF; no son implementación
```

Las carpetas `datos_crudos/`, `data/` y `cache/` son destinos de trabajo descritos en scripts; no se encontraron como carpetas de trabajo en la raíz de esta copia. El dato local de prueba está en `Context/DATA/`. Las rutas de Khipu y las rutas Windows guardadas en metadatos no indican la ubicación actual de este repositorio.

La fuente principal incluye introducción, capítulos 1–3 y anexos. Las inclusiones de capítulo 4, conclusiones y trabajos futuros están comentadas (`LATEX/asesoria5/main.tex:101–120`). Esos archivos existen, pero no son prueba de resultados obtenidos: contienen tablas vacías y marcadores como «por determinar» (`secciones/capitulo4.tex:31–69`; `secciones/conclusiones.tex:20–35`).

## 4. Módulos y conexiones reales

El diseño de cinco módulos está descrito en `LATEX/asesoria5/secciones/capitulo2.tex:816–1002`. **Las dos ramas de descriptores se encuentran en la fusión; los módulos no constituyen una sola cadena lineal 1→2→3→4→5.**

| Módulo | Función prevista | Evidencia y estado local |
|---|---|---|
| 1. Preprocesamiento | LAS → matriz S con XYZ, RGB y normales | Implementado en `preprocesar.py`; existe salida de PL7 y metadatos coherentes. |
| 2. Semántica | S → descriptor `f_sem` de 96 valores | Existe MinkUNet34C y entrenamiento base; falta adaptar el lector a S de la catedral. |
| 3. Encoder Sonata | S → descriptor `f_ssl`, caché por sección | `extraer_features.py` acepta el NPY y programa la inferencia; no se encontró caché de PL7 ni medición de esta extracción sobre PL7. |
| 4. Fusión | Alinear ambas ramas y sustituir `f_geo` por `f_ssl` | Existe razonamiento base de HyperNCD, pero no se encontró el consumidor del caché Sonata ni la sustitución integrada. |
| 5. Salida | Etiquetas, métricas y plataforma de consulta/exportación | Hay cálculo de IoU/mIoU en HyperNCD base; no se encontró plataforma ni salida final del híbrido sobre la catedral. |

Esto coincide con la tabla de estado de la propia tesis (`capitulo2.tex:976–1002`). Los porcentajes de esa tabla describen **aporte previsto**, no porcentaje de implementación completada.

### La rama semántica y el HyperNCD que sí existen

`hyperncd/train.py:13–15` usa `modules/Discoverer_hyper.py`, con lectores seleccionados para SemanticKITTI o SemanticPOSS (`hyperncd/utils/dataset.py:20–26`). El lector de SemanticPOSS lee binarios de cuatro columnas, conserva XYZ y construye características constantes iguales a 1 (`dataset.py:519–526`); después cuantiza con MinkowskiEngine (`dataset.py:637–647`). **No lee la matriz S de nueve columnas.**

`hyperncd/models/multiheadminkunet.py:353–407` construye el encoder y las cabezas de prototipos. La variante `MinkUNet34C` termina con 96 canales (`hyperncd/models/minkunet.py:206–207`).

En el recorrido activo de entrenamiento hay agrupamiento DBSCAN (`dataset.py:555–558`), promediado de características por región (`Discoverer_hyper.py:58–70`, `418–431`), descriptores geométricos manuales basados en autovalores y centros (`371–425`), afinidad semántica y geométrica, selección de vecinos y agregación (`448–476`), y generación de pseudoetiquetas con Sinkhorn (`479–484`). En ese bloque los nodos son **regiones**. No debe deducirse el recorrido activo únicamente de otras funciones llamadas «hypergraph» en `multiheadminkunet.py`.

Detalle para una futura implementación: `apply_gap_hypergraph` tiene valores locales `M=8`, `alpha=0.5`, `tau=0.1` y se llama sin reemplazarlos (`Discoverer_hyper.py:448`, `475–476`). El `self.hparams.alpha` que multiplica la pérdida regional en la línea 507 no está conectado a ese `alpha` local. Por tanto, el barrido de parámetros propuesto necesita revisar qué parámetro del código está variando realmente.

## 5. Módulo 1: explicación exacta de preprocesar.py

### 5.1 Entrada, parámetros y lectura por bloques

La entrada puede ser **uno o varios LAS**, porque `--input` usa `nargs="+"` (`preprocesar.py:35–36`). El programa interpreta todos los archivos pasados en esa llamada como una sección. No registra ni alinea escaneos entre sí: presupone un mismo sistema de coordenadas, tal como indica su cabecera (`13`).

| Parámetro | Valor por defecto | Uso |
|---|---:|---|
| `--voxel` | 0.05 | Lado del vóxel final, interpretado en metros |
| `--q` | 20 | Vecindad solicitada para SOR y normales |
| `--lam` | 2.0 | `std_ratio` del SOR |
| `--chunk` | 5 000 000 | Puntos por bloque de lectura LAS |
| `--prevoxel` | 0.0 | Voxelización adicional antes del SOR; 0 la desactiva |

Referencias: `preprocesar.py:39–49`.

`leer_las` abre cada archivo con `laspy.open` y recorre `f.chunk_iterator(chunk)` (`53–61`). Usa `bloque.x`, `bloque.y`, `bloque.z`: las coordenadas escaladas de LAS, no directamente los enteros crudos `X`, `Y`, `Z`. Extrae solo XYZ y RGB. No conserva intensidad, clasificación, retornos ni tiempo GPS.

**La lectura por bloques no equivale a un procesamiento completo con memoria acotada por bloque.** Los bloques se guardan en listas y se concatenan todos (`55–63`). El centrado, el SOR y las demás operaciones trabajan sobre la sección completa en memoria. Con varios LAS se acumulan todos. Con el valor por defecto, PL7 necesitaría dos bloques: 5 000 000 y 3 590 904 puntos; esto se deduce de su conteo, pero `chunk` no está guardado en el JSON de la prueba.

### 5.2 Conversión de RGB

RGB se convierte a `float32` durante la lectura. Tras concatenar todos los archivos, si el máximo global supera 255, **todos** los colores se dividen entre 256 (`60–66`). Si no, se mantienen.

Ejemplo: un canal LAS de 32 768 pasa a 128. No es una normalización por el máximo de la nube, no divide entre 65 535 ni convierte directamente a `uint8`.

La regla es una detección práctica de escala, no una validación por archivo. Si se mezclaran archivos con escalas RGB diferentes, aplicaría la misma división global. Para un valor arbitrario de 65 535, dividir entre 256 produce 255.99609375: la implementación **no recorta** a 255. El rango nominal «0–255» de los comentarios no es una garantía estricta para cualquier LAS. En las 100 muestras de PL7 contrastadas, los valores originales sí son múltiplos de 256.

Al crear la nube Open3D, vuelve a dividir por 255 (`preprocesar.py:70–74`), porque sus colores se representan nominalmente en 0–1. Tras voxelizar, multiplica por 255 antes de guardar (`119–121`). Esos cambios de escala no son etiquetas ni características aprendidas.

### 5.3 Centrado de coordenadas

El código calcula, por cada eje:

```text
offset = (mínimo de todos los puntos + máximo de todos los puntos) / 2
XYZ_centrado = XYZ_original − offset
```

Referencias: `preprocesar.py:83–87`.

Es el **centro de la caja envolvente**, no el promedio de las coordenadas de los puntos. Se calcula sobre las coordenadas realmente leídas de toda la sección, antes de eliminar ruido. No usa los límites declarados en la cabecera LAS para obtenerlo. No rota, cambia de escala ni reproyecta la nube.

Las coordenadas se centran antes del guardado final en `float32`, lo que evita representar allí directamente las coordenadas grandes. El JSON conserva el desplazamiento en `offset_utm`. Para un representante del NPY se puede recuperar su posición geográfica mediante `XYZ_npy + offset_utm`. Eso recupera la posición del representante con la precisión guardada; no recupera los puntos eliminados o agrupados.

Después del filtrado la caja de la nube puede dejar de estar centrada exactamente en cero. Tampoco se garantiza que la media de sus puntos sea cero. El nombre `offset_utm` expresa la interpretación del proyecto; el programa no comprueba un código EPSG.

### 5.4 Prevoxelización opcional

Si `--prevoxel > 0`, llama a `voxel_down_sample` **antes** del SOR y registra `n_prevoxel` (`preprocesar.py:92–95`). Cambia el conjunto sobre el que se filtra ruido y se estiman las estadísticas. No es el orden principal descrito en la tesis. En el JSON de PL7, `prevoxel=0.0`, así que esta opción figura desactivada.

### 5.5 SOR: eliminación estadística de puntos aislados

SOR se delega a Open3D:

```python
nube.remove_statistical_outlier(nb_neighbors=args.q, std_ratio=args.lam)
```

Referencias: `preprocesar.py:97–104`.

La explicación del propio código y la tesis es: calcular una distancia media de vecindad por punto, obtener la media global `μ` y su desviación estándar `σ`, y descartar los puntos que superan `μ + λσ`. Los argumentos solicitados por defecto son `q=20` y `λ=2`. No se implementa esa estadística manualmente en este repositorio; los detalles internos del conteo de vecinos y del umbral quedan a la versión instalada de Open3D.

SOR usa la geometría. No comprueba si un punto pertenece realmente a una columna, muro o ruido del escáner. «Punto descartado por SOR» es una clasificación estadística, no una etiqueta de referencia que certifique ruido físico. El JSON guarda el número que queda (`n_sor`), pero el programa descarta los índices devueltos por Open3D al asignarlos a `_`.

### 5.6 Voxelización y punto representante

Después del SOR ejecuta `nube.voxel_down_sample(args.voxel)` (`preprocesar.py:106–109`). El valor por defecto es 0.05: cubos de 5 cm si las coordenadas están en metros.

El comportamiento descrito por el código y los documentos es un representante por vóxel ocupado, con promedio de posiciones y colores. Para `m` puntos en un vóxel:

```text
posición representante = suma de XYZ / m
color representante = suma de RGB / m
```

No elige necesariamente un punto original, el centro geométrico del cubo ni el punto más cercano al escáner. Un promedio puede quedar en una posición que ningún punto original tenía. RGB puede tener decimales.

El repositorio delega la construcción de la rejilla a Open3D: no fija manualmente un origen global compartido por secciones. Tampoco conserva una tabla «punto crudo → representante». El orden de filas no debe interpretarse como el orden original del LAS.

La reducción por vóxeles es distinta de quitar ruido: agrupa mediciones. Sin embargo, una reducción espacial sí puede perder detalle fino o mezclar superficies cercanas. La frase «no se pierde ninguna superficie» de `capitulo3.tex:212–217` debe entenderse como una intención descriptiva; los conteos y las muestras no prueban una garantía geométrica de ese alcance.

### 5.7 Cálculo y orientación de normales

Las normales se calculan **sobre los representantes después de la voxelización**, con:

```python
nube.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=args.q))
nube.orient_normals_to_align_with_direction([0.0, 0.0, 1.0])
```

Referencias: `preprocesar.py:111–115`.

La explicación PCA del proyecto es estimar la covarianza de la vecindad y tomar el autovector del menor autovalor: la dirección de menor dispersión aproxima la perpendicular a la superficie. El programa solicita 20 vecinos por defecto. Son vecinos buscados otra vez sobre la nube reducida; reutilizar el mismo número `q` del SOR **no significa reutilizar los mismos puntos vecinos**.

La orientación alinea el signo con el eje +Z: busca que `n_z ≥ 0`. No es una orientación hacia el escáner ni una propagación de orientación a través de superficies conectadas. En superficies verticales, este criterio por sí solo no determina una orientación exterior/interior coherente. Una normal indica dirección, no clase semántica.

### 5.8 Orden completo y archivos guardados

Orden real:

```text
leer todos los LAS por bloques → concatenar XYZ/RGB → ajustar escala RGB
→ calcular/restar centro de caja → crear nube Open3D con RGB/255
→ [prevoxel, solo si se pide] → SOR → voxel final
→ calcular normales → orientar a +Z
→ concatenar XYZ, RGB×255, normales → convertir a float32
→ guardar NPY → registrar duración → guardar JSON
```

La matriz del NPY tiene forma `(N, 9)`, con una fila por representante final y **todas las columnas en `float32`** (`preprocesar.py:119–124`).

| Índices NumPy, desde 0 | Columnas | Significado |
|---|---|---|
| `0:3` | X, Y, Z | Coordenadas centradas |
| `3:6` | R, G, B | Color promediado, escala nominal 0–255 |
| `6:9` | Nx, Ny, Nz | Normal estimada y orientada |

No contiene etiquetas, intensidad, identificadores de punto original ni descriptores Sonata. El NPY no guarda nombres de columnas: el orden se establece en el código y su lector (`extraer_features.py:35–50`).

El JSON guarda entradas, `voxel`, `q`, `lambda`, `prevoxel`, `offset_utm`, `n_crudo`, `n_sor`, `n_voxel`, duración, y `n_prevoxel` si corresponde (`preprocesar.py:80–124`). No guarda `chunk`, versiones de bibliotecas, hardware, hash de entrada ni tiempos separados por etapa. `preprocesar.py` **no escribe PLY ni CSV**: son productos auxiliares presentes para inspección.

Los tiempos impresos en cada etapa son acumulados desde `t0`, no duraciones individuales. `segundos` se toma después de `np.save` y antes de escribir el JSON; no es únicamente el tiempo del SOR ni de las normales (`79`, `88–115`, `122–124`).

## 6. Evidencia de PL7_SCN0001.las

### 6.1 Conteos, parámetros y duración

Fuente de los metadatos: `Context/DATA/PL7_SCN0001_preprocesado.json:1–18`. El procedimiento y los mismos resultados se describen en `LATEX/asesoria5/secciones/capitulo3.tex:190–268`.

| Dato | Valor | Nivel de comprobación |
|---|---:|---|
| Puntos originales | 8 590 904 | Confirmado en cabecera LAS y coincide con JSON |
| Después de SOR | 8 184 419 | Registrado en JSON y tesis; no se repitió SOR |
| Descartados por SOR | 406 485, 4.7316 % | Aritmética a partir de los conteos registrados |
| Representantes finales | 610 637 | Confirmado en cabecera NPY y declarado en PLY/JSON |
| Retención voxel respecto de SOR | 7.4610 % | Aritmética a partir de metadatos |
| Retención final respecto del original | 7.1079 % | Aritmética sobre conteos |
| Factor de reducción original/final | 14.0688 | Aritmética sobre conteos |
| Vóxel final | 0.05 m | Registrado en JSON |
| Vecinos `q` | 20 | Registrado en JSON |
| Umbral `lambda` | 2.0 | Registrado en JSON |
| Prevoxel | 0.0 | Registrado como desactivado |
| Tiempo | 14.4 s | Registrado; no vuelto a medir |
| Forma y tipo de salida | `(610637, 9)`, `float32` | Confirmados en cabecera NPY |

Offset registrado:

```text
[278800.9714135999, 8667752.423552454, 174.09519999999998]
```

Se leyó este desplazamiento del JSON. Su igualdad con el centro de los extremos reales figura en el reporte previo (`Context/inspeccion_DATA.json:363`); no se volvieron a recorrer todos los puntos para comprobar los extremos. No hay que confundirlo con los offsets de codificación del LAS, que son otros números.

### 6.2 Comprobaciones ligeras realizadas ahora

1. **LAS:** se leyeron 227 bytes de cabecera. Firma `LASF`, versión 1.2, formato de punto 3, registros de 34 bytes, inicio de puntos en byte 227 y 8 590 904 puntos. Su tamaño real, **292 090 963 bytes**, coincide con `227 + 8590904 × 34`. Las escalas XYZ son 0.0001. La cabecera declara cero VLR; no se confirmó allí un EPSG, zona UTM, datum o referencia vertical. «UTM» y la procedencia del escáner GLS-2000 vienen de la documentación, no de una certificación geodésica en esta revisión.
2. **NPY:** cabecera versión 1.0, `descr='<f4'`, `fortran_order=False`, forma `(610637, 9)`, datos desde byte 128. Tamaño real **21 983 060 bytes**, exactamente `128 + 610637 × 9 × 4`. Son unos 21.98 MB decimales o 20.96 MiB; así se entiende que distintos documentos digan 21 o 22 MB.
3. **PLY:** solo se leyeron sus 266 bytes de cabecera. Declara 610 637 vértices, XYZ y normales `double`, colores `uchar`, formato binario little-endian. Tamaño real **31 142 753 bytes**, coherente con `266 + 610637 × 51`. Esto verifica su estructura declarada, no la igualdad completa de sus vértices con el NPY.
4. **CSV original:** 100 filas y diez columnas. Se compararon con 100 registros LAS repartidos a índices `floor(i × (N−1)/99)`, para `i=0..99`. Diferencia máxima por redondeo de XYZ menor de 0.00005; los otros campos comparados coinciden. En esos 100 puntos la clasificación es 0 y los colores son múltiplos de 256.
5. **CSV preprocesado:** 100 filas y nueve columnas, comparadas con 100 filas NPY a esos índices repartidos. Todas las diferencias son menores de 0.00005. En esos 100 registros las normales tienen longitud aproximada 1 (entre 0.999999961 y 1.000000040) y `n_z` positivo (mínimo aproximado 0.0020356).

Los 100 puntos crudos y los 100 representantes **no son parejas antes/después**. Se muestrearon sus respectivos archivos por posición de fila; comparar la fila 1 de un CSV con la fila 1 del otro no muestra la transformación de un mismo punto.

### 6.3 Evidencia previa que se leyó, pero no se regeneró

`Context/inspeccion_DATA.json` es un reporte preexistente. Declara que toda la clasificación LAS es cero, que el NPY completo es finito, que sus normales tienen longitud cercana a uno y `n_z` positivo (`47–170`, `188–332`). Sus extremos globales de RGB del NPY son R 0–238, G 0–226, B 0–251. **Estas comprobaciones globales no se repitieron ahora**: las comprobaciones nuevas se limitan a cabeceras y muestras.

También reporta una reproducción NumPy de la rejilla Sonata de 2 cm con **604 775 celdas distintas**, y declara explícitamente `model_executed=false` (`inspeccion_DATA.json:334–343`). No es una salida de inferencia de Sonata ni prueba de 604 775 descriptores ya extraídos. Muestra además que una rejilla de 2 cm puede volver a agrupar representantes de la voxelización previa de 5 cm: no se debe prometer conservación exacta del número de puntos entre las dos rejillas.

El guion `Context/inspeccionar_data.py` no se ejecutó: aunque usa memoria mapeada, calcula estadísticas sobre todos los puntos y reproduce la rejilla (`64–109`), además de otras comparaciones. No cumple la restricción de esta revisión de limitarse a lecturas pequeñas si se ejecuta completo.

### 6.4 Tiempo, equipo y corrida completa documentados

La tesis atribuye los 14.4 s a una laptop AMD Ryzen 5 8645HS, 16 GB de RAM, Windows 11, Python 3.12.7, laspy 2.7.0, NumPy 2.3.5 y Open3D 0.20.0. Indica que la RTX 3050 no intervino (`capitulo3.tex:70–80`, `259`). Esta información es **documentada**; no está certificada por el JSON de PL7, que no guarda el equipo o las versiones.

La tesis también declara una ejecución en Khipu sobre **23 LAS**, **589 708 290 puntos**, unos **20.05 GB**, que produjo **17 807 859 representantes** en **1 079 s** acumulados, con los mismos conteos de PL7 (`capitulo3.tex:174–181`, `264–268`). No se encontraron en esta copia los 23 JSON de salida ni el log de esa corrida: no se verificaron esos totales ni los tiempos.

La figura de un cubo de 1 m se describe con 48 098 puntos antes y 921 representantes después (`capitulo3.tex:234–243`). Existe la imagen y su guion (`LATEX/asesoria5/imagenes/figura_entrada_salida.py`), pero no se recalculó ese recorte. El guion también ajusta el brillo para visualizar (`45–52`); esa imagen no sirve para medir por sí sola la fidelidad numérica RGB.

## 7. Recorrido desde LAS hasta los resultados finales

### Tramo disponible y siguiente etapa programada

```text
LAS de una sección
  → preprocesar.py [CPU]
  → S.npy + metadatos.json
  → extraer_features.py [GPU, entorno Sonata]
  → caché .npz programado, sin archivo de PL7 encontrado
```

`preprocesar.sbatch:32–40` hace una llamada por cada LAS de `datos_crudos/`. Aunque el programa Python admite varios LAS por sección, el lanzador actual trata cada archivo como una sección independiente. No implementa por sí solo un agrupamiento espacial de varios escaneos. Omite una salida si ya existe su NPY salvo que se defina `REHACER` (`34–36`).

`extraer_features.py:35–50` acepta `.npy` o texto de nueve columnas y separa `coord`, `color`, `normal`. La ayuda y la cabecera todavía hablan principalmente de `.txt`; la compatibilidad NPY sí está en el código.

Carga pesos de Sonata con `sonata.load`, pasa el modelo a CUDA y activa evaluación (`54–60`); usa `torch.inference_mode()` (`112–117`). Por tanto, esta etapa extrae representaciones y no entrena el encoder.

La transformación de Sonata (`sonata/sonata/transform.py:1205–1224`) aplica:

1. Otro desplazamiento: centro de la caja en X/Y y mínimo en Z (`182–195`). Es distinto del centrado XYZ del Módulo 1.
2. `GridSample` de 0.02 m, configurable con `--grid` en el extractor (`extraer_features.py:98–104`). Selecciona puntos por celda, en lugar de promediarlos como el Módulo 1.
3. RGB dividido entre 255 (`transform.py:153–157`).
4. Conversión a tensores y concatenación de coordenadas, colores y normales como entrada del modelo (`67–90`, `1216–1222`).

El `GridSample` predeterminado usa `mode="train"` y selección aleatoria por celda (`transform.py:846–851`, `1212`), incluso aunque el modelo esté en evaluación. El extractor no fija una semilla. No hay que trasladar a este tramo una promesa de determinismo completo del preprocesamiento.

`upcast` concatena dos niveles y propaga los restantes por los índices del pooling (`extraer_features.py:63–82`). Devuelve características a la resolución de la **rejilla de Sonata**. Después todavía hace falta `feat[inverse]` para asignar un vector a cada fila de S; el README local de Sonata muestra ese paso expresamente (`sonata/README.md:178–180`).

El NPZ programado guarda (`extraer_features.py:119–142`):

| Clave | Forma / tipo previsto en el código | Significado |
|---|---|---|
| `feat` | `(N_rejilla, D)`, `float16` por defecto o `float32` con `--fp32` | Descriptores |
| `coord` | `(N_rejilla, 3)`, `float32` | Coordenadas de esa rejilla, después del desplazamiento Sonata |
| `inverse` | `(N_S,)`, `int32` | Para cada fila de S, índice de su descriptor en `feat` |
| `grid_size` | escalar `float32` | Tamaño de rejilla |
| `n_original` | escalar `int64` | Número de filas de S leído por el extractor, no puntos LAS crudos |

La dimensión **1088** es la declarada por el diseño (`capitulo3.tex:288–308`); el código imprime la dimensión obtenida, pero no la impone (`extraer_features.py:129`). El modelo se construye con la configuración del checkpoint (`sonata/sonata/model.py:758–792`), así que no se presenta aquí como dimensión medida en PL7.

### Tramo pendiente del híbrido

```text
S.npy ──→ lector de catedral pendiente ──→ HyperNCD/MinkUNet ──→ f_sem
  └─────→ Sonata ──→ NPZ ──→ consumidor/alineamiento pendiente ──→ f_ssl
                 ambas ramas → fusión pendiente → hipergrafo → etiquetas
                 → evaluación de catedral → plataforma/exportación propuestas
```

No se encontró código que cargue el NPZ de Sonata en el entrenamiento de HyperNCD. Aún hay que alinear índices y rejillas; el `upcast` por sí solo no resuelve esa integración. Además, el NPZ no conserva explícitamente el desplazamiento adicional de `CenterShift` ni copia `offset_utm`: para interpretar sus coordenadas en el mapa se necesita mantener la relación con S y sus metadatos. `inverse` ofrece una correspondencia con S, no con todos los puntos crudos del LAS.

Los dos entornos (`sonata-env` y `hyperncd-env`) y el intercambio por disco son la estrategia documentada para las pilas de dependencias diferentes (`README.md:21–42`). No se comprobó su instalación o compatibilidad en la máquina actual.

## 8. Qué resultados existen y qué falta

**Implementado y con artefactos de prueba:** Módulo 1 sobre PL7, NPY/JSON, muestras pequeñas y documentación. La coherencia de cabeceras y muestras no prueba por sí sola que cada representante sea exactamente el promedio correcto o que la corrida se haya hecho con la versión actual: no se guardaron hashes de procedencia y no se reprodujo el procesamiento.

**Implementado como código, sin prueba local del híbrido completo:** extracción Sonata a NPZ; entrenamiento y evaluación de HyperNCD en datasets base; DBSCAN, descriptores manuales, agregación por afinidades y Sinkhorn.

Hay un registro pequeño `hyperncd/logs/default/version_154/metrics.csv` con 76 filas de datos. Su última fila con validación contiene `epoch=1`, `step=1243`, `valid/mIoU=0.34740012884140015` (aproximadamente 34.74 %). El archivo asociado `hparams.yaml:8–9`, `38`, `45` declara SemanticPOSS, `pretrained: null`, `split: 0`. **Es evidencia guardada de una corrida base, no resultado de PL7 ni del híbrido.** No se comprobó quién produjo ese registro, su procedencia, la ejecución completa de las diez épocas configuradas ni su reproducibilidad.

Los logs `logs/demo-sonata_50849.out:21–29` muestran carga de pesos y del ejemplo `sample1`, seguida de problemas para crear una ventana gráfica. `logs/demo-sonata_50840.err:162–183` registra un fallo de acceso/ausencia de caché de Hugging Face. No son medidas de extracción de features de la catedral ni resultados de su segmentación.

**Pendiente para completar la tesis según el diseño y el código encontrado:**

- Adaptar el lector de HyperNCD a S y definir secciones, etiquetas y particiones de la catedral. La anotación pendiente está reconocida en `capitulo2.tex:979–982`.
- Ejecutar y medir la extracción Sonata sobre las secciones: número retenido, dimensión efectiva, tiempo, memoria y tamaño de caché. La tabla de `capitulo3.tex:293–314` aún contiene valores «por medir» y una cota estimada de memoria.
- Implementar el consumidor del caché, la alineación y la sustitución `f_geo → f_ssl` en el recorrido activo del hipergrafo.
- Medir la línea base y el híbrido en datos comparables, clases conocidas/novedosas y repeticiones. No se encontró una demostración local de mejora mIoU en la catedral.
- Calibrar los parámetros reales del modelo; implementar/evaluar el análisis y la ablación previstos. No presentar las tablas vacías del capítulo 4 como mediciones.
- Implementar la plataforma de cortes, selección de elementos y exportación 3D descrita en `capitulo2.tex:947–974`. No se encontró una aplicación que lo haga.
- Si se requiere una etiqueta para cada punto LAS original, diseñar el mapeo correspondiente: el Módulo 1 actual no conserva esa relación.

Otras discrepancias documentales: `conclusiones.tex:44–46` habla de 1–2 GB y cinco secciones, mientras `capitulo3.tex:174–181` describe 23 LAS y 20.05 GB. No se verificó un agrupamiento final en cinco secciones. Tampoco debe entenderse «misma salida» de `capitulo3.tex:250–262` como identidad binaria garantizada entre versiones de Open3D: el código no incorpora aleatoriedad explícita en el Módulo 1, pero no se compararon corridas, ordenamientos o entornos.

## 9. Archivos pequeños que conviene compartir con otro ChatGPT

### Paquete mínimo para empezar por el Módulo 1

1. `EXPLICACION_PARA_CHATGPT.md` — esta guía.
2. `preprocesar.py` — implementación completa, unos 5.3 KB.
3. `Context/DATA/PL7_SCN0001_preprocesado.json` — parámetros y conteos, 332 bytes.
4. `Context/DATA/PL7_muestra_100.csv` — estructura original, unos 7.2 KB.
5. `Context/DATA/PL7_preprocesado_muestra_100.csv` — estructura final, unos 6.6 KB.
6. `Context/DATA/README.md` — descripción del dato, unos 3.7 KB.

### Para explicar la tesis y el flujo de ambos modelos

Añadir `README.md`, `extraer_features.py`, `preprocesar.sbatch`, `extraer_features.sbatch` y estas fuentes:

- `LATEX/asesoria5/secciones/introduccion.tex` — problema y objetivos.
- `LATEX/asesoria5/secciones/capitulo2.tex` — fundamentos y cinco módulos.
- `LATEX/asesoria5/secciones/capitulo3.tex` — método, prueba y pendientes.
- `Context/inspeccion_DATA.json` — reporte anterior; recordar su condición de evidencia guardada.

Para una lectura más técnica de HyperNCD/Sonata, añadir solo los archivos que se vayan a estudiar: `hyperncd/modules/Discoverer_hyper.py`, `hyperncd/models/multiheadminkunet.py`, `hyperncd/models/minkunet.py`, `hyperncd/utils/dataset.py`, `hyperncd/utils/callbacks.py`, `sonata/sonata/transform.py` y `sonata/README.md`. `Context/inspeccionar_data.py` es opcional para explicar el reporte anterior, sin necesidad de ejecutarlo.

### Exclusiones concretas

No compartir los `.las`, `.laz`, NPY/PLY completos, `Context/DATA.zip`, datasets, pesos de modelos, checkpoints, cachés `.npz`/`.pkl`, entornos virtuales, dependencias instaladas, `__pycache__`, `.git`, `.aws`, `.codex`, archivos `.env`, claves SSH, tokens ni registros completos de W&B. Para esta explicación tampoco hacen falta todos los PDF de referencia ni los archivos auxiliares de compilación LaTeX.

**Excluir los originales `hyperncd/train.py` y `hyperncd/train_KITTI.py`: contienen asignaciones literales de `WANDB_API_KEY` en las líneas 81 y 83, respectivamente.** No se reproduce su contenido sensible en esta guía. Si después se necesita estudiar esas entradas, compartir únicamente una copia revisada con la credencial eliminada. Esta revisión no modificó esos archivos. No basta con excluir carpetas de credenciales: también hay que considerar secretos incrustados en código.

## 10. Instrucción sugerida para el siguiente chat

> Explícame esta tesis en español, paso a paso, usando la guía y los archivos adjuntos. Empieza por qué representa un punto LAS y después recorre preprocesar.py en su orden real. En cada paso explica entrada, operación, motivo, salida y líneas del código. Usa los conteos de PL7 distinguiendo lo documentado de lo comprobado. No supongas que la fusión HyperNCD–Sonata ya está terminada ni que existe una mejora mIoU demostrada. Después explica la extracción Sonata y lo que falta conectar. Divide la explicación en lecciones y comprueba mi comprensión antes de avanzar.
