# Modelos y pesos: cuál es el correcto y dónde está

Para no confundirse. Cada pieza dice **qué es, dónde está y cómo comprobar que es la buena**. Lo que no se pudo
comprobar está marcado. Drive = `PFC1_HyperNCD_Sonata/` en tu Mi unidad.

## 1. M2 · HyperNCD (`f_sem`)

| | |
|---|---|
| **Checkpoint correcto (el único que se usa)** | `split0_run4/SemanticPOSS_S0_best-epoch=8-step=11196.ckpt` |
| Ruta completa en Drive | `m2/entrada/datos_m2_descomprimidos/hyperncd/checkpoints/SemanticPOSS/split0_run4/SemanticPOSS_S0_best-epoch=8-step=11196.ckpt` |
| Qué es | HyperNCD entrenado en SemanticPOSS (partición 0); mejor época de la corrida `run4`, mIoU de validación **0,3837** |
| Cómo se comprobó | el `.json` de M2 de la escena nueva dice `epoch: 8`, `global_step: 11196` (comprobado el 9-nov-2026) |

**Otros checkpoints que hay en esa carpeta (NO usar):**

| Carpeta | Archivo | Por qué no |
|---|---|---|
| `split0_full` | `best-epoch=0-step=1244`, `last` | casi sin entrenar: casi todos los puntos caían en una sola cabeza |
| `split0_run3` | `best-epoch=1-step=2488`, `last` | corrida anterior, menos entrenada |
| `split0_run4` | `best-epoch=4-step=6220` | época intermedia |
| `split0_run4` | `last.ckpt`, `last-v1/v2/v3` | último guardado, no el mejor |

**Cómo comprobar que una salida de M2 usó el bueno:** abre su `.json` (`<escena>_fsem.json`) y mira `ckpt` y `ckpt_info`:

```python
import json
j = json.load(open("<ruta>_fsem.json")); print(j["ckpt"]); print(j["ckpt_info"])   # debe decir epoch 8, global_step 11196
```

Los cuadernos nuevos (`colab_1_m1_m2.ipynb`) exigen ese archivo exacto en modo `AUTO` y comprueban la época cargada.

**Salidas de M2 (`f_sem`):**

| Qué | Dónde |
|---|---|
| PL7 y PL13, **época 8** (la buena) | local `cache/PL7_fsem.npz`, `cache/PL13_fsem.npz`; Drive `m2/salida/resultados_m2/` |
| PL7 y PL13, época 0 (descartada, solo para comparar) | local `cache/m2_epoch0/` |
| Escena etiquetada `listo` (época 8, verificada) | Drive `m2/salida/capilla_etiquetada/capilla_etiquetada_fsem.npz` |

## 2. M3 · Sonata (`f_ssl`)

### 2a. Sonata **original** (de Meta)

| | |
|---|---|
| Qué es | pesos preentrenados `sonata.pth` del repositorio `facebook/sonata` en Hugging Face. No hay archivo propio |
| Cómo se carga | `extraer_features.py` sin `--pesos`; en el log aparece `Loading checkpoint from HuggingFace: sonata` |
| `f_ssl` original de PL13 | local `experimentacion/datos/modulo3_PL13/PL13_feats.npz` (sacado en Khipu el 19-oct-2026) |
| Escena etiquetada | Drive `m3/salida/capilla_etiquetada/capilla_etiquetada_feats_original.npz` |

### 2b. Sonata **adaptado** a la Catedral (el que entrenamos)

| | |
|---|---|
| **Pesos correctos** | Drive `m3/pesos/sonata_catedral.pth` (formato de `sonata.load`) |
| De dónde salen | exportados el 8-nov-2026 desde `m3/entrenamiento/model/model_last.pth` (**época 30**), tomando el *teacher* |
| Cómo se entrenó | 21 escaneos (todos **menos PL7 y PL13**), 30 épocas, `LOOP=5`, **lote 1** (105 iteraciones por época, 3 150 en total), LR 0,0002, 32 768 puntos por vista, bf16, atención SDPA, GPU L4. Sin etiquetas |
| Cómo se comprobó | la celda de exportación imprimió "453/453 tensores vienen del entrenamiento" y **cambio relativo de pesos 8,35 %** |
| `f_ssl` adaptado de PL7 y PL13 | Drive `m3/salida/PL7_SCN0001_adaptado_feats.npz`, `PL13_SCN0001_adaptado_feats.npz`; local `experimentacion/datos/modulo3_PL7/` y `modulo3_PL13/PL13_feats_adaptado.npz` |
| Escena etiquetada | Drive `m3/salida/capilla_etiquetada/capilla_etiquetada_feats_adaptado.npz` |

**Checkpoints de entrenamiento** (`m3/entrenamiento/model/`, 1,81 GB cada uno): `epoch_5`, `10`, `15`, `20`, `25`, `30` y `model_last`
(= época 30). Sirven para exportar otra época y compararla. **No hay "mejor"**: sin etiquetas no hay métrica para elegirla.
El registro de pérdida está en `m3/entrenamiento/train.log`.

**Cómo comprobar que una extracción usó el adaptado:** en el log de `extraer_features.py` debe aparecer
`Loading checkpoint in local path: .../m3/pesos/sonata_catedral.pth`. El cuaderno 2 también imprime, antes de usar la GPU, el tamaño y
la fecha del archivo de pesos (debe ser del 8-nov-2026).

**No usar:** los resultados de la **primera corrida de adaptación** (10 iteraciones por época por un error de `loop`; ~300 iteraciones en total).
Si en Drive ves una carpeta `model_prueba_10it` o un `train_prueba_10it.log`, son de esa corrida. (No pude ver tu Drive para confirmar si existen.)

## 3. Entornos (no son pesos, pero también se confunden)

| Entorno | Archivo en Drive (`entornos/`) |
|---|---|
| HyperNCD (M2) | `hyperncd_env_cache.tar.gz` |
| Sonata (M3, M4) | `pointcept_env_py313_torch2.14.1_cu130.tar.gz` (hoy compilado para la **L4**, arquitectura 8.9); los nuevos llevan `_sm<GPU>` |

## 4. Qué no está comprobado

- **La GPU con la que M2 funcionó antes.** Los resultados antiguos no guardan su nombre. Desde ahora `predecir_hyperncd.py` lo guarda en el `.json`.
- **El contenido actual de tu Drive.** No puedo listarlo desde aquí; lo de esta página sale de tus cuadernos, de los `.json` de resultados y de lo que me pegaste.
- Si `sonata_catedral.pth` es la versión de la época 30 es lo que dicen la fecha y el log de exportación, no una comparación de contenido.

## 5. Regla práctica

Antes de interpretar un resultado, comprueba **tres líneas**: la de `ckpt`/`epoch` en el `.json` de M2, la de `Loading checkpoint ...` en el log de M3
y la fecha del archivo de pesos adaptados. Si alguna no coincide con esta página, no uses ese resultado.
