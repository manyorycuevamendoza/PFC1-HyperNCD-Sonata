# Experimento: Sonata original vs Sonata adaptado a la Catedral

**Pregunta:** ¿cambia y mejora el descriptor `f_ssl` (Módulo 3) si se continúa el preentrenamiento
auto-supervisado de Sonata con escaneos de la Catedral? ¿Qué cuesta y cómo se ve después en el
Módulo 4?

**Estado:** hecho y comparado **sin etiquetas** (solo PL13). Todavía no se sabe si el adaptado
*acierta más*: eso necesita el recorte etiquetado (`../etiquetado/`) y un control, ver "Qué falta".

## Qué se hizo

| Paso | Detalle |
|---|---|
| Datos de entrenamiento | los 21 escaneos de `data/` **menos PL7 y PL13** (se dejan fuera para evaluar). Sin etiquetas |
| Entrenamiento | Pointcept, config `pretrain-sonata-v1m1-0-base` modificada. Parte de `sonata.pth` (Meta) y sigue con el mismo objetivo auto-supervisado. 30 épocas × 105 iteraciones = 3 150 (`LOOP=5`, **lote 1**, 32 768 puntos por vista, LR 0,0002, bf16; datos tomados de la salida guardada del cuaderno en Drive) |
| Dónde corrió | Google Colab, GPU L4 de 22,5 GB (cuaderno `colab_sonata_adaptacion.ipynb`). La versión para Khipu está en `sonata_adaptacion/` y todavía no se corrió |
| Atención | parche SDPA de PyTorch (no hay flash-attn); equivalente matemáticamente a la atención original |
| Pesos usados | el *teacher* de la época 30 (`model_last.pth`), exportado a `m3/pesos/sonata_catedral.pth` |
| Extracción | `extraer_features.py --pesos sonata_catedral.pth` sobre PL7 y PL13 |

## Qué hay en esta carpeta

```
adaptacion_sonata/
├── README.md                 esta guía
├── comparar_m3.py            compara f_ssl original vs adaptado (PL13)
├── comparar_m4.py            compara el Módulo 4 con cada f_ssl
├── costo.py                  gráfico y tabla de costo computacional
├── visor_adaptacion.py       visor 3D (.html) original vs adaptado, M3 y M4
├── datos_costo/
│   └── costo_medido.csv      cada cifra de costo con su fuente (los vacíos son PENDIENTE)
└── resultados/
    ├── m3/                   comparacion_m3_PL13.json y .png
    ├── m4_original/          Módulo 4 con f_ssl ORIGINAL   (json + grupos)
    ├── m4_adaptado/          Módulo 4 con f_ssl ADAPTADO   (json + grupos)
    ├── m4_comparacion/       tabla_m4.md, comparacion_m4_PL13.json y .png
    ├── costo/                costo_computacional.png
    └── visor/                visor_adaptacion_PL13.html (NO va a git; se regenera con visor_adaptacion.py)
```

## Cómo reproducirlo (desde la raíz del repo, en la laptop, sin GPU)

Necesita estos archivos (no van a git; ver "Dónde están los datos"):
`experimentacion/datos/modulo3_PL13/{PL13_SCN0001.npy, PL13_feats.npz, PL13_feats_adaptado.npz}`,
`experimentacion/datos/modulo3_PL7/PL7_feats_adaptado.npz` y `cache/PL13_fsem.npz`.

```bash
python experimentacion/adaptacion_sonata/comparar_m3.py        # ~2 min
python experimentacion/modulo4_real/correr_modulo4_real.py \
    --npy experimentacion/datos/modulo3_PL13/PL13_SCN0001.npy --fsem cache/PL13_fsem.npz \
    --fssl experimentacion/datos/modulo3_PL13/PL13_feats.npz \
    --salida experimentacion/adaptacion_sonata/resultados/m4_original
python experimentacion/modulo4_real/correr_modulo4_real.py \
    --npy experimentacion/datos/modulo3_PL13/PL13_SCN0001.npy --fsem cache/PL13_fsem.npz \
    --fssl experimentacion/datos/modulo3_PL13/PL13_feats_adaptado.npz \
    --salida experimentacion/adaptacion_sonata/resultados/m4_adaptado
python experimentacion/adaptacion_sonata/comparar_m4.py
python experimentacion/adaptacion_sonata/costo.py
python experimentacion/adaptacion_sonata/visor_adaptacion.py   # visor 3D, ~1 min
```

**Visor 3D:** abre `resultados/visor/visor_adaptacion_PL13.html` en el navegador (necesita internet
para cargar Plotly). Botones: color real, PCA y k-means de cada descriptor, mapa de cuánto cambió
cada punto, y los grupos del Módulo 4 (K = 4 y 8) con cada `f_ssl`. Los grupos del adaptado se
reordenan para que el mismo color sea, en lo posible, el mismo grupo del original.

El Módulo 4 original reproduce exactamente las cifras de `../modulo4_real/README.md`
(estabilidad de M3 0,629 y de M4 0,552 con K = 4).

## Resultados

### Módulo 3: cuánto cambió el descriptor (PL13, 200 000 puntos originales)

| Medida | Original | Adaptado |
|---|---|---|
| Cambio relativo de los pesos del *teacher* (norma) | | **8,35 %** (453 de 453 tensores cambiaron) |
| Coseno punto a punto (centrado) | media **0,54** (p5 0,35) | |
| Varianza de la PCA (3 / 10 comps.) | 0,253 / 0,434 | 0,225 / 0,477 |
| Rango efectivo (riqueza del espacio) | 110,5 | **81,3** |
| k-means K = 8, estabilidad entre semillas | 0,61 | 0,49 |
| ARI de k-means original vs adaptado | **0,21** | |

El descriptor adaptado es **bastante distinto** del original y algo menos "rico" (rango efectivo
más bajo). Figura: `resultados/m3/comparacion_m3_PL13.png`.

### Módulo 4 (PL13, sin etiquetas)

| K | Variante | Estabilidad orig. | Estabilidad adapt. | ARI orig~adapt |
|---|---|---|---|---|
| 4 | M3 | 0,629 | 0,567 | 0,248 |
| 4 | M4 | 0,552 | 0,493 | 0,331 |
| 8 | M3 | 0,628 | 0,684 | 0,249 |
| 8 | M4 | 0,705 | 0,607 | 0,336 |

La estabilidad **baja en tres de los cuatro casos** y sube solo en M3 con K = 8. No hay
evidencia de que el adaptado mejore los grupos. Tabla completa en
`resultados/m4_comparacion/tabla_m4.md`. M2 y M4 base no usan `f_ssl` y dan lo mismo en ambas
corridas (control).

### Costo computacional

`resultados/costo/costo_computacional.png`. Entrenamiento: unos **103 min** en la L4
(3 150 iteraciones: 105 por época, ~1,9 s por iteración, ~3,5 min por época) y 13,3 GB de checkpoints. Además hubo una corrida
inválida de ~42 min por un error de configuración (`loop`), que cuenta como costo de depuración.
Extraer `f_ssl` de PL13 con el original costó 13,9 s y 34,6 GB de VRAM en Khipu (RTX A6000, atención densa);
con el adaptado costó **20,3 s y 7,40 GB** en la L4 con atención SDPA (PL7: 15,0 s y 5,73 GB), según la salida guardada
del cuaderno. **No son comparables entre sí**: cambian la GPU y el tipo de atención. Sirven para decir que el adaptado
cabe en una L4.

## Cómo leer estos resultados (límites)

- **Sin etiquetas no se puede decir qué descriptor acierta más.** Solo se mide cambio y estabilidad.
- **Falta un control.** El original se extrajo en Khipu (sin semilla fija, atención densa) y el
  adaptado en Colab (semilla 0, atención SDPA). La rejilla de PL13 tiene 796 315 filas en uno y
  796 309 en el otro. Parte del cambio (coseno 0,54) podría venir de esa diferencia de
  entorno y no de la adaptación.
  **Control propuesto:** extraer PL13 en Colab con los pesos **originales** (`sonata.pth`) y el
  mismo parche y semilla, y medir el coseno contra el original de Khipu. Si sale cerca de 1, el
  cambio es por la adaptación; si sale cerca de 0,54, es por el entorno.
- Una sola corrida de entrenamiento (una semilla, un LR). No hay barras de error.
- PL7 solo tiene el adaptado (no existe su `f_ssl` original aquí), así que no se compara.

## Qué falta

1. **Control** de arriba (extraer con pesos originales en el mismo entorno).
2. **Pegar la salida de la celda 8** de Colab (`>>> TIEMPO`, `>>> VRAM PICO`) y completar las
   filas PENDIENTE de `datos_costo/costo_medido.csv`; luego `python .../costo.py`.
3. **`train.log`** de `m3/entrenamiento/` en `datos_costo/train.log`: `costo.py` lo usa solo y
   agrega el tiempo real por época.
4. **mIoU contra el recorte etiquetado de PL13** para M3 y M4, original contra adaptado.
5. Probar otras épocas guardadas (`epoch_10`, `epoch_20`, ...) y otro LR.

## Dónde están los datos

| Qué | Dónde |
|---|---|
| `f_ssl` adaptado de PL7 y PL13 | Drive `m3/salida/`; copia local en `experimentacion/datos/modulo3_PL7/` y `modulo3_PL13/` |
| Pesos adaptados (`sonata_catedral.pth`) | Drive `m3/pesos/` |
| Checkpoints de entrenamiento y `train.log` | Drive `m3/entrenamiento/` |
| `f_ssl` original de PL13 | `experimentacion/datos/modulo3_PL13/PL13_feats.npz` (generado en Khipu) |

Todo el árbol de Drive está explicado en `documentacion/drive_indice.md`.
