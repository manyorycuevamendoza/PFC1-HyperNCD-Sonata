# Estado del proyecto (actualizado el 6 de noviembre de 2026)

Mapa de qué módulos están hechos, dónde está el código, los datos y los
resultados de cada uno, y qué falta. Para el detalle de cada módulo, ver los
documentos enlazados.

## Resumen por módulo

| Módulo | Qué hace | Estado | Código | Resultados |
|---|---|---|---|---|
| **M1** Preprocesamiento | SOR, voxelización 5 cm, normales | Hecho (los 23 escaneos en Khipu; PL7 y PL13 de prueba) | `preprocesar.py`, `preprocesar.sbatch` | `experimentacion/resultados/informe/` |
| **M2** HyperNCD | Descriptor semántico `f_sem` y descubrimiento de clases | Línea base en SemanticPOSS y **`f_sem` ya extraído en PL7 y PL13** con el checkpoint de la época 8 (Colab). Ver "M2 en Colab" | `hyperncd/`, `predecir_hyperncd.py`, `colab_runner.ipynb` | `experimentacion/resultados/modulo2_hyperncd/`, `cache/PL*_fsem.npz` |
| **M3** Sonata | Descriptores `f_ssl` (1088 dim.) con encoder congelado | Hecho (PL13 en Khipu) | `extraer_features.py`, `extraer_features.sbatch`, `sonata/` | `experimentacion/resultados/modulo3_PL13.json` y `.png` |
| **M4** Fusión e hipergrafo | `F = [f_sem ; f_ssl]`, regiones, prototipos, Sinkhorn | Simulado **y corrido sobre PL13 real** (sin etiquetas aún: solo comparación entre variantes). Ver "M4 real" | `experimentacion/modulo4_simulado/`, `experimentacion/modulo4_real/` | `experimentacion/resultados/modulo4_real/` |
| **M5** Evaluación | mIoU sobre la Catedral | No empezado | — | — |

Fuente del estado de M1–M5: `Context/CONTEXTO_DEFENSA_MODULOS_1_Y_3.md`
(sección 1). Si algo cambia, actualiza esta tabla.

## Dónde está cada cosa

### Módulo 1 y Módulo 3 (PL7 y PL13)

- **Código:** `preprocesar.py` (M1), `extraer_features.py` (M3).
- **Documentación:** `Context/CONTEXTO_DEFENSA_MODULOS_1_Y_3.md` (procedimiento
  paso a paso, cifras y código completo) y `experimentacion/README.md`
  (cómo generar las figuras y qué datos van con cuál).
- **Escaneos completos:** no están en git ni en esta máquina. Viven en Khipu,
  en `~/PFC1-HyperNCD-Sonata/datos_crudos/` (los 23 `.las`). El `.npy` y el
  `.npz` de PL13 están en Khipu en `data/` y `cache/`.
- **Muestras locales:** `Context/DATA/` (100 puntos de PL7 en CSV y su JSON).
- **Visor 3D:** `experimentacion/visor_3d.py`. Genera un `.html` por escaneo
  (no van a git) y se exportan PNG con el botón de cámara del propio visor.
- **Figuras para la tesis:** `experimentacion/resultados/informe/`
  (`PL7_*.png`, `PL13_*.png`, `m1_paso*.png`, `m3_paso1_entrada.png` y las
  cuatro `lamina_*.png`). Se componen con
  `experimentacion/figuras_informe/componer_lamina.py`.
- **Medidas de Sonata:** `experimentacion/resultados/modulo3_PL13.json`
  (generado por `experimentacion/modulo3_sonata/visualizar_features.py`).
- **Paquete para compartir:** `Context/paquete_chatgpt/` (contexto, código,
  imágenes, medidas y borrador de la tesis) y su `.zip`.

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
`PFC1_HyperNCD_Sonata/datos_m2_descomprimidos/hyperncd/checkpoints/SemanticPOSS/`.

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

### Etiquetado de un recorte de PL13 (en curso)

La Catedral no tiene etiquetas (el `.las` trae `classification = 0`). Para medir mIoU se
etiqueta a mano un recorte de ±7 m en el centro de PL13 (214 645 puntos) con CloudCompare.
Archivos y pasos en `experimentacion/etiquetado/` (`LEEME_etiquetar.md`); hay
pre-etiquetas automáticas en `prelabel/` para que solo se corrijan. El paquete para pasar
a quien etiquete es `etiquetado_PL13.zip`. Las etiquetas devueltas
(`clase_N_*.ply`, con el campo `idx`) se leen contra
`PL13_recorte_indices.npy`. Los `.ply` y `.npy` no van a git.

## Pendiente

1. **Etiquetar el recorte de PL13** (en curso, ver arriba) y calcular mIoU y matriz de
   confusión de cada variante de M4. Es lo que falta para poder comparar M2, M3 y M4 de verdad.
2. **Probar más fusiones de M2 y M3 en M4:** pesos por rama, PCA de `f_ssl` a 96 dim.,
   fusión tardía de afinidades, confianza de M2 por región, dos hipergrafos.
3. **M2 sobre el resto de escaneos** (hoy solo PL7 y PL13) y mejorar M2 en la Catedral
   (el modelo viene de calles).
4. **Documentar** la línea base de HyperNCD: mIoU por clase y comparación con el paper.
5. **Pruebas de parámetros** de M1 y M3 (P0 a P5). El procedimiento está en
   `experimentacion/PROMPT_PRUEBAS_PARAMETROS.md`, pero no existe todavía la
   carpeta `experimentacion/pruebas_parametros/` ni sus scripts
   (`medir_calidad.py`, `comparar_features.py`). Necesitan los escaneos completos y la GPU de Khipu.
6. **M5**: evaluación sobre la Catedral.

## Advertencias para no perder trabajo

- El `.npz` de Sonata va con **su** `.npy`: los descriptores se asignan por el
  orden de las filas. Ver la regla en `experimentacion/README.md`.
- Hay tres carpetas con nombre parecido en `~/Documentos`:
  `PFC1-HyperNCD-Sonata-1` (esta, con todo el código, los logs y los
  checkpoints), `PFC1-HyperNCD-Sonata-main` (parcial, 74 archivos) y
  `GitHub/PFC1-HyperNCD-Sonata` (clone con `.git`, sin los archivos del
  proyecto). Ninguna de las dos primeras es un repositorio git.
