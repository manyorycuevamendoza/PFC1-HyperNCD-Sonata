# Entrenar HyperNCD (M2) con ArCH

**Pregunta.** ¿Un M2 entrenado con patrimonio arquitectónico (ArCH) deja de colapsar en la capilla? Y cuánto tiempo, RAM y GPU cuesta entrenarlo.

| Archivo | Para qué |
|---|---|
| `convertir_arch.py` | `.txt` de ArCH a bloques `.npz` de 10 m: voxel 5 cm, etiqueta por mayoría de votos, reanudable |
| `../../hyperncd/config/arch_dataset.yaml` | clases, `other` ignorada, clases nuevas (`novel_classes`) |
| `../../hyperncd/utils/dataset.py` (`ArCHRestrictedDataset`) | lee los bloques y devuelve lo mismo que el dataset de SemanticPOSS |
| `../../hyperncd/train.py` | acepta `--dataset ArCH` y `--novel` |
| `../monitor_recursos/monitor_recursos.py` | mide tiempo, RAM y GPU de cada comando |
| `../../colab_entrenamiento_arch.ipynb` | convierte, prueba 1 época medida y entrena la corrida completa |

**Decisiones (cambiables)**
- Entrenamiento = carpeta `Training` (15 escenas); validación = `Test` (2). Reparto oficial de ArCH.
- Clases: 9 (`other` ignorada). Nuevas: `arch`, `column`, `vault`. Conocidas: `moldings`, `floor`, `door_window`, `wall`, `stairs`, `roof`.
- Solo geometría (features = unos), como en SemanticPOSS. Se entrena desde cero, igual que la corrida original.
- Muestra = bloque de 10 m, 60 000 puntos al azar por época; las regiones de la clase nueva (DBSCAN) se calculan sobre el bloque completo y se cachean.

**Flujo en Colab** (`colab_entrenamiento_arch.ipynb`): convierte ArCH, entrena **30 épocas** (reanudable), predice la **capilla etiquetada** con el mejor checkpoint y con el último, y empaqueta logs, predicciones y mediciones en un zip. Las métricas de dominio (cabeza dominante, NMI, ARI, mIoU 1 a 1) se calculan en local con `../m2_dominio/medir_m2.py --fsem capilla=<npz>,<etiquetas.npy>,<etiquetas.json>` (ya acepta modelos con 9 cabezas).

**Compatibilidad con Colab:** el entorno trae torchmetrics 0.7.2 y pytorch-lightning 1.4.8. `train.py` y `Discoverer_hyper.py` tienen respaldo para eso; la celda 2b del cuaderno los aplica sola si GitHub aún no los trae. Con `num_workers=2` el DataLoader se colgaba, por eso `WORKERS = 0`.

## Resultados (corrida en Colab T4, 10-nov-2026; una corrida, una semilla)

**Costo del entrenamiento (30 épocas):** 75 min (2,5 min por época), RAM de proceso pico 4,9 GB (sistema 5,4 de 12,7 GB), GPU pico 9,5 GB de 15,4 GB (uso medio 63 %), CPU 75 %. Predecir la capilla (1,14 M de puntos): 43 s, 4,5 GB de RAM, 3,2 GB de VRAM. Hay margen para subir el batch.

**Validación en ArCH** (2 escenas `Test`, mismo dominio del entrenamiento): mIoU **0,489** en la época 20 (mejor) y 0,485 en la 30. Por clase (mejor): roof 0,89, wall 0,80, vault* 0,72, floor 0,68, column* 0,56, moldings 0,34, stairs 0,21, arch* 0,16, door_window 0,05 (*novedosas). Curvas: `resultados/entrenamiento/curvas.png`.

**Capilla etiquetada** (otro dominio; mismas métricas que `../m2_dominio/`, calculadas en local con `medir_m2.py`):

| M2 | Cabeza dominante | Cabezas > 1 % | NMI | NMI sin «otro» | mIoU 1 a 1 | mIoU 1 a 1 sin «otro» |
|---|---|---|---|---|---|---|
| SemanticPOSS, época 8 (el anterior) | **98,4 %** | 1 | 0,018 | 0,027 | 0,114 | 0,077 |
| ArCH, mejor (época 20) | 44,5 % | 9 | 0,192 | 0,437 | 0,179 | 0,336 |
| ArCH, última (época 30) | 44,6 % | 9 | 0,191 | 0,439 | 0,183 | 0,355 |

Gráficos: `resultados/m2_capilla/m2_dominio.png`, `m2_cabezas.png`, `m2_cabeza_x_clase.png`; tabla completa en `resultados/m2_capilla/resumen.md` y `resultados.json`.

**Qué se concluye**
1. **El colapso de M2 en la capilla se debía al dominio de entrenamiento, no al pipeline.** Con el mismo código de inferencia y las mismas métricas, entrenar con patrimonio (ArCH) reparte los puntos en las 9 cabezas y sube el NMI de 0,02 a 0,19 (0,44 sin `otro`). Esto confirma la hipótesis de `../m2_dominio/` con un control causal parcial.
2. **El mejor checkpoint y el último son casi iguales** (época 20 contra 30): la validación se aplana, no hace falta entrenar más con esta receta.
3. **Advertencia sobre `f_sem` y M4:** en la capilla solo el **12,6 %** de los puntos cae en las 3 cabezas novedosas (con SemanticPOSS era ~99 %). M4 solo agrupaba los puntos novedosos, así que con este M2 agruparía una fracción pequeña y no sería comparable con lo anterior. Se añadió `--todos` en `../modulo4_real/correr_modulo4_real.py` para agrupar todos los puntos; el M4 de la capilla con este M2 se corre con `colab_celda_m4.py` (abajo).
4. El rango efectivo de `f_sem` es bajo (1,7), pero **no es señal de colapso** (ver `../m2_dominio/`): aquí lo que importa es el reparto de cabezas y el NMI.

**Límites.** Una corrida y una semilla. ArCH incluye iglesias y patrimonio, pero la capilla sigue siendo otro edificio y otro sensor (la clase `arch` de ArCH ya cuesta: IoU 0,16 en su propia validación). El NMI y el mIoU 1 a 1 usan las etiquetas de la capilla para emparejar cabezas con clases; mide cuánto separan las cabezas, no un clasificador entrenado con esas clases. No se midió si M2-ArCH mejora M4: eso está pendiente.

## Cómo reproducir las métricas en local (laptop, sin GPU)

```bash
# 1. el zip que baja la celda 9 del cuaderno: descomprimirlo en experimentacion/arch_entrenamiento/resultados_colab/
# 2. curvas y resumen del entrenamiento
python experimentacion/arch_entrenamiento/graficar_entrenamiento.py
# 3. métricas de M2 en la capilla, comparando el M2 anterior con el de ArCH
python experimentacion/m2_dominio/medir_m2.py --salida experimentacion/arch_entrenamiento/resultados/m2_capilla \
  --etiqueta semposs_e8_vs_arch \
  --fsem "capilla_SemanticPOSS_e8=<fsem_viejo.npz>,<etiquetas.npy>,<etiquetas.json>" \
  --fsem "capilla_ArCH_best_e21=experimentacion/arch_entrenamiento/resultados_colab/predicciones/capilla_etiquetada_best_fsem.npz,<etiquetas.npy>,<etiquetas.json>" \
  --fsem "capilla_ArCH_last_e30=experimentacion/arch_entrenamiento/resultados_colab/predicciones/capilla_etiquetada_last_fsem.npz,<etiquetas.npy>,<etiquetas.json>"
# <etiquetas.*> = experimentacion/escena_capilla_etiquetada/datos/m1/salida/capilla_etiquetada/capilla_etiquetada_etiquetas.{npy,json}
# <fsem_viejo.npz> = experimentacion/escena_capilla_etiquetada/datos/m2/salida/capilla_etiquetada/capilla_etiquetada_fsem.npz
# 4. registrar y regenerar el visor/índice
python experimentacion/registro/registrar.py
```

## M4 con el M2 de ArCH (corrido en la laptop, CPU, 79 s y 68 s)

M4 con el `f_sem` de ArCH (último checkpoint) y los `f_ssl` de la capilla (Sonata original y adaptado), con `--todos` (agrupa todos los puntos; ver conclusión 3). mIoU en %, **sin `otro`, K = 8, asignación 1 a 1** (entre paréntesis, por mayoría). Mismo evaluador y mismas etiquetas que la corrida anterior. Tabla y gráfico: `resultados/m4_capilla/comparacion_m2.{md,csv,png}`.

| Método | M2 SemanticPOSS + Sonata original | **M2 ArCH** + Sonata original | M2 SemanticPOSS + Sonata adaptado | **M2 ArCH** + Sonata adaptado |
|---|---|---|---|---|
| M2 (solo `f_sem`) | 34,1 (34,2) | **49,0** (49,1) | 34,1 (34,2) | **49,0** (49,1) |
| M3 (solo `f_ssl`) | 53,4 (48,2) | 52,5 (47,3) | 35,6 (37,3) | 35,1 (35,0) |
| M4 base (`f_sem` + `f_geo`) | 32,1 (29,9) | **52,9** (50,6) | 32,1 (29,9) | **52,9** (50,6) |
| M4 (`f_sem` + `f_ssl`) | 52,5 (48,9) | 50,6 (50,8) | 36,1 (33,4) | **47,4** (40,6) |

**Qué se concluye (con una sola corrida, una escena y una semilla: indicios, no demostraciones)**
1. **Un M2 sano cambia el resultado.** Solo `f_sem`: de 34 a 49. La línea base M4 base: de 32 a 53. M3 no cambia (no depende de M2).
2. **Cambia la lectura anterior de M4.** Antes, "reemplazar `f_geo` por `f_ssl` sube ~20 puntos" (52,5 contra 32,1). Con el M2 de ArCH, M4 base (52,9) alcanza a M4 (50,6) y a M3 solo (52,5): la ventaja de `f_ssl` sobre `f_geo` **dependía de que `f_sem` estuviera colapsado**. Con la asignación por mayoría M4 queda apenas arriba (50,8 contra 50,6) y en K = 12 también (51,9 contra 51,0), pero son diferencias que una sola semilla no sostiene.
3. **`f_sem` ahora sí aporta**, pero M4 no mejora a M3 ni a M4 base. Esto hay que contarlo tal cual en la tesis: la propuesta no supera a la línea base cuando el M2 funciona (en esta escena).
4. **El Sonata adaptado sigue siendo peor** que el original en M3 (35,1 contra 52,5), con cualquier M2. Con el M2 de ArCH, M4 adaptado mejora (36,1 a 47,4) por la parte de `f_sem`, no por la adaptación.
5. **Clases finas:** arco y columna siguen siendo las peores (columna IoU ~0 en M3 y M4 con K = 6; ver `resultados/m4_capilla/evaluacion_sin_otro/evaluacion.md`).

**Límites.** Con `--todos` se agrupan todos los puntos (la salida de `evaluar_m4.py` aún dice "solo los que M2 marca como novedosos": ya no aplica). El M2 de ArCH se entrenó con otras clases conocidas/novedosas que las de la capilla. Falta: más escenas y semillas, M4 con otras fusiones, y la sonda lineal de Sonata.

**Visor 3D** (M1 a M4, aciertos y errores, original contra adaptado): `visor/visor_escena_m2_arch.html` (no va a git; se regenera con `generar_visor_escena.py --datos arch_entrenamiento/datos_visor --resultados arch_entrenamiento/resultados/visor_base`). No lo abrí en un navegador, solo comprobé que se genera.

**Cómo se corrió M4 en local** (`f_ssl` bajados de Drive a `escena_capilla_etiquetada/datos/m3/salida/capilla_etiquetada/`):
```bash
python experimentacion/modulo4_real/correr_modulo4_real.py --npy <capilla_etiquetada.npy> \
  --fsem experimentacion/arch_entrenamiento/resultados_colab/predicciones/capilla_etiquetada_last_fsem.npz \
  --fssl <capilla_etiquetada_feats_original|adaptado.npz> --salida experimentacion/arch_entrenamiento/resultados/m4_capilla/m4_<variante> --todos --k 6 8 12
python experimentacion/evaluacion_etiquetada/evaluar_m4.py --etiquetas <..._etiquetas.npy> --info <..._etiquetas.json> \
  --grupos original=<m4_original>/grupos_PL13.npz adaptado=<m4_adaptado>/grupos_PL13.npz --salida experimentacion/arch_entrenamiento/resultados/m4_capilla/evaluacion_sin_otro --ignorar otro
python experimentacion/arch_entrenamiento/comparar_m2_m4.py
```
`colab_celda_m4.py` hace lo mismo en Colab (CPU) si no se tienen los `f_ssl` en la laptop.

## Qué hay en esta carpeta

| Ruta | Qué es | ¿Va a git? |
|---|---|---|
| `convertir_arch.py`, `graficar_entrenamiento.py`, `comparar_m2_m4.py`, `colab_celda_m4.py` | código | sí |
| `resultados/entrenamiento/` | curvas, resumen y medición de recursos del entrenamiento | sí |
| `resultados/m2_capilla/` | métricas e imágenes de M2 en la capilla (SemanticPOSS contra ArCH) | sí |
| `resultados/m4_capilla/` | evaluaciones de M4 (`evaluacion/`, `evaluacion_sin_otro/`), `comparacion_m2.{md,csv,png}` y logs; los `grupos_PL13.npz` no van a git | sí, salvo los `.npz` |
| `resultados/visor_base/` | tablas estilo papers, etapas y evaluaciones que usa el visor | sí |
| `visor/visor_escena_m2_arch.html`, `datos_visor/` | visor 3D y copias de trabajo (enlaces duros a los datos) | no (se regeneran) |
| `resultados_colab/` | el zip de Colab tal cual (logs, mediciones y predicciones `.npz`) | solo lo liviano; los `.npz` no |

Licencia de ArCH: CC BY-NC-SA 4.0 (Politecnico di Torino); citar el dataset en la tesis.
