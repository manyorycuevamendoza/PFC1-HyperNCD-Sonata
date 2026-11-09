# Estado del proyecto (actualizado el 8 de noviembre de 2026)

Mapa de qué módulos están hechos, dónde está el código, los datos y los
resultados de cada uno, y qué falta. Para el detalle de cada módulo, ver los
documentos enlazados.

## Resumen por módulo

| Módulo | Qué hace | Estado | Código | Resultados |
|---|---|---|---|---|
| **M1** Preprocesamiento | SOR, voxelización 5 cm, normales | Hecho (los 23 escaneos en Khipu; PL7 y PL13 de prueba) | `preprocesar.py`, `preprocesar.sbatch` | `experimentacion/resultados/informe/` |
| **M2** HyperNCD | Descriptor semántico `f_sem` y descubrimiento de clases | Línea base en SemanticPOSS y **`f_sem` ya extraído en PL7 y PL13** con el checkpoint de la época 8 (Colab). Ver "M2 en Colab" | `hyperncd/`, `predecir_hyperncd.py`, `colab_runner.ipynb` | `experimentacion/resultados/modulo2_hyperncd/`, `cache/PL*_fsem.npz` |
| **M3** Sonata | Descriptores `f_ssl` (1088 dim.) con encoder congelado | Hecho con Sonata original (PL13 en Khipu). **Adaptación a la Catedral: en curso**, ver "Adaptación de Sonata" | `extraer_features.py`, `extraer_features.sbatch`, `sonata_adaptacion/`, `colab_sonata_adaptacion.ipynb`, `sonata/` | `experimentacion/resultados/modulo3_PL13.json` y `.png` |
| **M4** Fusión e hipergrafo | `F = [f_sem ; f_ssl]`, regiones, prototipos, Sinkhorn | Simulado **y corrido sobre PL13 real** (sin etiquetas aún: solo comparación entre variantes). Ver "M4 real" | `experimentacion/modulo4_simulado/`, `experimentacion/modulo4_real/` | `experimentacion/resultados/modulo4_real/` |
| **M5** Evaluación | mIoU sobre la Catedral | No empezado | — | — |

Fuente del estado de M1–M5: `documentacion/contexto/CONTEXTO_DEFENSA_MODULOS_1_Y_3.md`
(sección 1). Si algo cambia, actualiza esta tabla.

## Mapa de carpetas

```
raíz/
├── ESTADO_DEL_PROYECTO.md, README.md, SETUP_KHIPU.md   guías (empieza por esta)
├── preprocesar.py / .sbatch        M1  preprocesamiento (corre en Khipu)
├── predecir_hyperncd.py            M2  inferencia de HyperNCD (la usa colab_runner.ipynb)
├── hyperncd_*.sbatch               M2  entrenamiento e instalación en Khipu
├── extraer_features.py / .sbatch   M3  f_ssl con Sonata (acepta --pesos adaptados)
├── demo_sonata.sbatch              M3  demo de Sonata
├── sonata_adaptacion/              M3  adaptación auto-supervisada de Sonata en Khipu
├── colab_runner.ipynb              M2  en Google Colab
├── colab_sonata_adaptacion.ipynb   M3  adaptación de Sonata en Google Colab
├── colab_pipeline_etiquetado.ipynb M1→M4  pipeline completo con evaluación con etiquetas (escena `listo`)
├── experimentacion/                M4, visores, etiquetado y resultados (cada subcarpeta con su README)
│   ├── registro/ y visor/          registro de experimentos (.json) y visor con pestañas por etapa y modo comparar
├── hyperncd/, sonata/              código de terceros (clonados; no tocar)
├── documentacion/                  contexto/ (muestras de datos y defensa), tesis_latex/, papers/, prompts/, historico/
└── data/, cache/, logs/            datos de trabajo y salidas de Slurm (no van a git)
```

Los scripts de Khipu y los cuadernos de Colab se quedan en la raíz a propósito: sus rutas
(`sbatch extraer_features.sbatch`, `python extraer_features.py ...`) están escritas en las
guías y en los cuadernos. Moverlos exige cambiar todo eso a la vez.

## Dónde está cada cosa

### Módulo 1 y Módulo 3 (PL7 y PL13)

- **Código:** `preprocesar.py` (M1), `extraer_features.py` (M3).
- **Documentación:** `documentacion/contexto/CONTEXTO_DEFENSA_MODULOS_1_Y_3.md` (procedimiento
  paso a paso, cifras y código completo) y `experimentacion/README.md`
  (cómo generar las figuras y qué datos van con cuál).
- **Escaneos completos:** no están en git ni en esta máquina. Viven en Khipu,
  en `~/PFC1-HyperNCD-Sonata/datos_crudos/` (los 23 `.las`). El `.npy` y el
  `.npz` de PL13 están en Khipu en `data/` y `cache/`.
- **Muestras locales:** `documentacion/contexto/DATA/` (100 puntos de PL7 en CSV y su JSON).
- **Visor 3D:** `experimentacion/visor_3d.py`. Genera un `.html` por escaneo
  (no van a git) y se exportan PNG con el botón de cámara del propio visor.
- **Figuras para la tesis:** `experimentacion/resultados/informe/`
  (`PL7_*.png`, `PL13_*.png`, `m1_paso*.png`, `m3_paso1_entrada.png` y las
  cuatro `lamina_*.png`). Se componen con
  `experimentacion/figuras_informe/componer_lamina.py`.
- **Medidas de Sonata:** `experimentacion/resultados/modulo3_PL13.json`
  (generado por `experimentacion/modulo3_sonata/visualizar_features.py`).

### Módulo 4 simulado

`experimentacion/modulo4_simulado/` (ver su `README.md`). Corre en la laptop
sin GPU en unos 3 minutos y compara la línea base (`f_geo`) con la propuesta
(`f_ssl`) sobre una escena sintética.

### HyperNCD: línea base en SemanticPOSS (partición 0)

Corrida local del 4 de noviembre de 2026, en la ThinkStation (no en Khipu).

- **Comando:** `hyperncd/tools/entrenar_resiliente.sh split0_run4 -s 0
  --dataset SemanticPOSS --offline --num_workers 0 --batch_size 2`. El script
  reanuda desde `last.ckpt` si el entrenamiento se cae.
- **Retomó** desde el checkpoint de `split0_run3`. 10 épocas (0 a 9).
- **Mejor validación:** mIoU **0.38370** en la época 8 (época 6: 0.37835;
  época 7: 0.38117; la época 9 no lo superó).
- **Checkpoint a usar:**
  `hyperncd/checkpoints/SemanticPOSS/split0_run4/SemanticPOSS_S0_best-epoch=8-step=11196.ckpt`
- **Log:** `hyperncd/logs/train_split0_run4.out`.
- **Incidente:** a las 16:07 se cayó una vez, en la época 6, con
  `ValueError: NumPy boolean array indexing assignment cannot assign 55040
  input values to the 55947 output values where the mask is true`. El
  reintento automático lo resolvió. La causa no se investigó.
- **Sin completar:** mIoU por clase y comparación con el valor del paper.

### M2 en Colab (PL7 y PL13)

El `f_sem` de la Catedral se saca en **Google Colab** con `colab_runner.ipynb`
(MinkowskiEngine no corre en la laptop). Los pesos están en Drive, en
`PFC1_HyperNCD_Sonata/m2/entrada/datos_m2_descomprimidos/hyperncd/checkpoints/SemanticPOSS/`.
  Organización del Drive: una carpeta por módulo (`m1`, `m2`, `m3`, `m4`, cada una con
  `entrada` y `salida`) y `entornos/` para los entornos compilados. Las celdas de
  `colab_runner.ipynb` y `colab_sonata_adaptacion.ipynb` crean esa estructura y acomodan los
  archivos viejos sin pisar nada.

- **Checkpoint correcto:** `split0_run4/SemanticPOSS_S0_best-epoch=8-step=11196.ckpt`
  (mIoU de validación 0.38370). Hay que ponerlo **a mano** en `CKPT_PATH` de la celda 5.
  Con `"AUTO"` el cuaderno agarraba `split0_full ... epoch=0` (casi sin entrenar) y
  casi todos los puntos caían en una sola clase.
- **Salidas:** `PL7_fsem.npz` y `PL13_fsem.npz` (96 dim. por vóxel, `inverse` al `.npy`).
  Se copian a `cache/` (no van a git). Las copias del modelo de la época 0 están en
  `cache/m2_epoch0/`.
- **Resultados en el repo:** `experimentacion/resultados/modulo2_hyperncd/`
  (`*_prediccion_hyperncd.json`, `modulo2_*.json/.png`; los `.html` no van a git).
- **Lectura honesta:** con la época 8, en PL13 el 82 % de los puntos sigue en la
  cabeza novedosa 0 y el 13 % en la 12. Mejoró, pero sigue concentrado: el modelo se
  entrenó en calles (SemanticPOSS), no en una catedral.

### M4 real (PL13)

`experimentacion/modulo4_real/` (ver su `README.md`). Une `f_sem` (M2) y `f_ssl` (M3)
por punto original y corre las fórmulas del M4 sobre regiones de 0,75 m. Solo PL13 tiene
M2 **y** M3 a la vez (PL7 no tiene `f_ssl` aquí). Variantes: M2, M3, M4 base
`[f_sem ; f_geo]` y M4 `[f_sem ; f_ssl]`.

Resultado (K = 4, `f_sem` de la época 8):

| | Estabilidad (ARI entre semillas) | Acuerdo con M4 (ARI) |
|---|---|---|
| M2 | 0.53 | 0.17 |
| M3 | 0.63 | 0.46 |
| M4 base | 0.64 | 0.14 |
| M4 | 0.55 | — |

Con `f_sem` de la época 0, M4 era casi una copia de M3 (ARI 0.87, estabilidad 0.95); con
la época 8 mezcla de verdad las dos fuentes pero es menos estable. Esos números viejos
están en `experimentacion/resultados/modulo4_real_m2_epoch0/`.
**Sin etiquetas no se puede decir qué variante acierta más.**
Visor 3D: `experimentacion/resultados/modulo4_real/visor_modulo4_PL13.html`.

### Adaptación de Sonata a la Catedral (hecha; falta evaluarla con etiquetas)

Objetivo: continuar el preentrenamiento auto-supervisado de Sonata (sin etiquetas) con los 21
escaneos de la Catedral **menos PL7 y PL13**, sacar el `f_ssl` adaptado de PL7 y PL13 y compararlo
contra el Sonata original con el recorte etiquetado de PL13. Se hace con Pointcept
(`configs/sonata/pretrain-sonata-v1m1-0-base.py`), no con la carpeta `sonata/` del repo (solo inferencia).

- **Colab (L4, 22,5 GB):** la primera corrida fue inválida por un error de `loop` (10 iteraciones por época). Corregido,
  se reentrenó: 30 épocas × 52 iteraciones, ~103 min, `f_ssl` adaptado de PL7 y PL13 extraído (con parche SDPA).
- **Khipu:** `sonata_adaptacion/` (preparar + sbatch en un solo job). Escrito, **sin correr todavía**. Comandos en `SETUP_KHIPU.md`.
- **Resultado (PL13, sin etiquetas)** — detalle y figuras en `experimentacion/adaptacion_sonata/`:
  el descriptor adaptado es bastante distinto del original (coseno medio 0,54; rango efectivo 110 → 81; ARI de
  k-means 0,21). En el Módulo 4 la estabilidad baja en 3 de 4 casos (M3/M4 con K = 4 y 8) y sube solo en M3 con K = 8.
  **No hay evidencia de mejora**, pero tampoco se puede decir que empeora: sin etiquetas no se mide acierto.
- **Falta un control:** el original se extrajo en Khipu y el adaptado en Colab (otra atención y semilla).
  Hay que extraer PL13 con los pesos originales en Colab para separar el efecto de la adaptación del del entorno.
- **Costo:** 103 min de L4 + 42 min de una corrida inválida; 13,3 GB de checkpoints. El costo de extracción del adaptado
  queda por medir (ver `experimentacion/adaptacion_sonata/README.md`).
- **Elegir la época:** no hay `model_best.pth` (sin etiquetas no hay métrica). Se guardaron épocas 5 a 30; se elige por mIoU en el recorte.

### Etiquetado de un recorte de PL13 (en curso)

La Catedral no tiene etiquetas (el `.las` trae `classification = 0`). Para medir mIoU se
etiqueta a mano un recorte de ±7 m en el centro de PL13 (214 645 puntos) con CloudCompare.
Archivos y pasos en `experimentacion/etiquetado/` (`LEEME_etiquetar.md`); hay
pre-etiquetas automáticas en `prelabel/` para que solo se corrijan. El paquete para pasar
a quien etiquete es `etiquetado_PL13.zip`. Las etiquetas devueltas
(`clase_N_*.ply`, con el campo `idx`) se leen contra
`PL13_recorte_indices.npy`. Los `.ply` y `.npy` no van a git.

## Escena etiquetada nueva (`listo`, 8 de noviembre de 2026)

19 `.las` con una pieza por clase (arco, columna, cúpula, pared, techo, otro): una escena de 3,38 M de puntos con
etiquetas reales. **No está dentro de PL7 ni PL13** (a ~10 m). Herramientas en `experimentacion/evaluacion_etiquetada/`,
cuaderno `colab_pipeline_etiquetado.ipynb` y guía `documentacion/ejecutar_pipeline_colab.md`. **Falta correrlo en Colab**
(M2 y M3 no se han probado con ella); con esto se podrá medir mIoU de M2, M3 y M4, original contra adaptado.

## Recordatorios para la tesis

- [ ] **Sobre-segmentación de superficies curvas en M4** (bóvedas partidas en varios grupos, mientras los anillos
  de la cúpula sí se separan bien). Está anotado en `experimentacion/modulo4_real/README.md`. Escribirlo como limitación
  observada y como motivación de la mejora de continuidad espacial.
- [ ] La adaptación de Sonata **no mostró mejora** sin etiquetas y falta el control con pesos originales
  (`experimentacion/adaptacion_sonata/README.md`). Reportar como resultado negativo/inconcluso, no omitirlo.
- [ ] Declarar el uso de herramientas de IA donde la universidad lo pida.

Todo lo que quedó abierto para Tesis 2 está reunido en `experimentacion/tesis2/README.md`.

## Ideas para mejorar M4 (por probar)

Orden de menor a mayor esfuerzo. Las referencias se citaron de memoria: **verificarlas antes de citarlas**.

1. **Medir el problema sin etiquetas.** *Fragmentación*: componentes conexos por grupo, y fracción de pares de regiones
   vecinas con grupos distintos. Da un número antes/después de cada mejora.
2. **Restricción de continuidad espacial (rápido).** Agrupar con conectividad: clustering jerárquico (Ward) restringido al
   grafo de regiones vecinas, o suavizar etiquetas sobre ese grafo (potencial de Potts con *graph cuts*, Boykov et al. 2001;
   CRF denso, Krähenbühl y Koltun 2011). Se suma al M4 como una arista de hipergrafo por adyacencia.
3. **Regiones que sigan la superficie en vez de celdas de 0,75 m.** Crecimiento de regiones con restricción de suavidad
   (Rabbani et al. 2006, ISPRS), segmentación por grafo (Felzenszwalb y Huttenlocher 2004), parches localmente convexos
   (LCCP, Stein et al. 2014) o supervóxeles (VCCS, Papon et al. 2013). Una bóveda queda en una sola región.
4. **Sobre-segmentar y luego fusionar** (superpoints): *Superpoint Graph* (Landrieu y Simonovsky, CVPR 2018) y
   *Superpoint Transformer* (Robert et al., ICCV 2023) agrupan en piezas geométricamente simples y razonan sobre sus relaciones.
5. **Descriptores menos sensibles a la orientación.** Quitar o ponderar menos las normales, o usar medidas de tipo de
   superficie (planar, cilíndrica) en vez de dirección.
6. **Jerarquía.** Dejar K alto y fusionar grupos vecinos parecidos hasta un umbral, en vez de fijar K de antemano.

## Pendiente

1. **Control de la adaptación** (extraer PL13 con pesos originales en Colab), completar el costo y **medir mIoU** original vs adaptado con el recorte etiquetado.
2. **Etiquetar el recorte de PL13** (en curso, ver arriba) y calcular mIoU y matriz de
   confusión de cada variante de M4. Es lo que falta para poder comparar M2, M3 y M4 de verdad.
3. **Probar más fusiones de M2 y M3 en M4:** pesos por rama, PCA de `f_ssl` a 96 dim.,
   fusión tardía de afinidades, confianza de M2 por región, dos hipergrafos.
4. **M2 sobre el resto de escaneos** (hoy solo PL7 y PL13) y mejorar M2 en la Catedral
   (el modelo viene de calles).
5. **Documentar** la línea base de HyperNCD: mIoU por clase y comparación con el paper.
6. **Pruebas de parámetros** de M1 y M3 (P0 a P5). El procedimiento está en
   `documentacion/prompts/PROMPT_PRUEBAS_PARAMETROS.md`, pero no existe todavía la
   carpeta `experimentacion/pruebas_parametros/` ni sus scripts
   (`medir_calidad.py`, `comparar_features.py`). Necesitan los escaneos completos y la GPU de Khipu.
7. **M5**: evaluación sobre la Catedral.

## Advertencias para no perder trabajo

- El `.npz` de Sonata va con **su** `.npy`: los descriptores se asignan por el
  orden de las filas. Ver la regla en `experimentacion/README.md`.
- Hay tres carpetas con nombre parecido en `~/Documentos`:
  `PFC1-HyperNCD-Sonata-1` (esta, con todo el código, los logs y los
  checkpoints), `PFC1-HyperNCD-Sonata-main` (parcial, 74 archivos) y
  `GitHub/PFC1-HyperNCD-Sonata` (clone con `.git`, sin los archivos del
  proyecto). Ninguna de las dos primeras es un repositorio git.
