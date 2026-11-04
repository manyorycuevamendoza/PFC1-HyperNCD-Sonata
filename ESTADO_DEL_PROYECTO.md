# Estado del proyecto (actualizado el 4 de noviembre de 2026)

Mapa de qué módulos están hechos, dónde está el código, los datos y los
resultados de cada uno, y qué falta. Para el detalle de cada módulo, ver los
documentos enlazados.

## Resumen por módulo

| Módulo | Qué hace | Estado | Código | Resultados |
|---|---|---|---|---|
| **M1** Preprocesamiento | SOR, voxelización 5 cm, normales | Hecho (los 23 escaneos en Khipu; PL7 y PL13 de prueba) | `preprocesar.py`, `preprocesar.sbatch` | `experimentacion/resultados/informe/` |
| **M2** HyperNCD | Descriptor semántico `f_sem` y descubrimiento de clases | Solo la línea base en SemanticPOSS (ver abajo). Falta sobre la Catedral | `hyperncd/`, `hyperncd_train_poss.sbatch` | `hyperncd/logs/`, `hyperncd/checkpoints/` |
| **M3** Sonata | Descriptores `f_ssl` (1088 dim.) con encoder congelado | Hecho (PL13 en Khipu) | `extraer_features.py`, `extraer_features.sbatch`, `sonata/` | `experimentacion/resultados/modulo3_PL13.json` y `.png` |
| **M4** Fusión e hipergrafo | `F = [f_sem ; f_ssl]`, regiones, prototipos, Sinkhorn | Solo simulado con datos sintéticos. Falta el real | `experimentacion/modulo4_simulado/` | `experimentacion/modulo4_simulado/resultados/` |
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

## Pendiente

1. **Documentar** la línea base de HyperNCD: mIoU por clase y comparación con
   el paper.
2. **Pruebas de parámetros** de M1 y M3 (P0 a P5). El procedimiento está en
   `experimentacion/PROMPT_PRUEBAS_PARAMETROS.md`, pero no existe todavía la
   carpeta `experimentacion/pruebas_parametros/` ni sus scripts
   (`medir_calidad.py`, `comparar_features.py`), así que no se han ejecutado.
   Necesitan los escaneos completos y la GPU de Khipu.
3. **M2 real** sobre la Catedral (`f_sem`).
4. **M4 real** con `f_sem` y `f_ssl` de los escaneos.
5. **M5**: evaluación.

## Advertencias para no perder trabajo

- El `.npz` de Sonata va con **su** `.npy`: los descriptores se asignan por el
  orden de las filas. Ver la regla en `experimentacion/README.md`.
- Hay tres carpetas con nombre parecido en `~/Documentos`:
  `PFC1-HyperNCD-Sonata-1` (esta, con todo el código, los logs y los
  checkpoints), `PFC1-HyperNCD-Sonata-main` (parcial, 74 archivos) y
  `GitHub/PFC1-HyperNCD-Sonata` (clone con `.git`, sin los archivos del
  proyecto). Ninguna de las dos primeras es un repositorio git.
