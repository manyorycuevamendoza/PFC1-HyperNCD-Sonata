# Índice del Google Drive del proyecto

Carpeta: **`PFC1_HyperNCD_Sonata`** (Mi unidad de `renzofelixaponte@gmail.com`). Una carpeta por módulo.
Los cuadernos de Colab crean y ordenan esta estructura solos (celda 1), sin pisar archivos.

```
PFC1_HyperNCD_Sonata/
├── LEEME_DRIVE.md                       este índice (copia de documentacion/drive_indice.md)
├── m1/
│   ├── entrada/capilla_etiquetada/      etiquetas crudas de la capilla (.npz)
│   └── salida/
│       ├── data_m1.tar.gz               carpeta data/ de Khipu (23 .npy + .json del Módulo 1)
│       └── capilla_etiquetada/          .npy preprocesado + _etiquetas.npy/.json (M1 de la escena etiquetada)
├── m2/
│   ├── entrada/datos_m2_descomprimidos/hyperncd/checkpoints/SemanticPOSS/
│   │       split0_run4/SemanticPOSS_S0_best-epoch=8-step=11196.ckpt   checkpoint de HyperNCD (SemanticPOSS)
│   └── salida/
│       ├── resultados_m2/               f_sem de PL7 y PL13 y resultados de M2 (colab_runner.ipynb)
│       └── capilla_etiquetada/          f_sem de la capilla con el M2 de SemanticPOSS (156 MB)
├── m3/
│   ├── entrenamiento/                   adaptación de Sonata (colab_sonata_adaptacion.ipynb)
│   │   ├── model/                       epoch_5, 10, 15, 20, 25, 30 y model_last.pth (1,8 GB c/u)
│   │   └── train.log                    registro de Pointcept (pérdida y tiempos por época)
│   ├── pesos/sonata_catedral.pth        pesos adaptados en formato sonata.load
│   └── salida/
│       ├── PL7_SCN0001_adaptado_feats.npz, PL13_SCN0001_adaptado_feats.npz
│       └── capilla_etiquetada/          f_ssl de la capilla: _feats_original.npz y _feats_adaptado.npz (~1 GB c/u comprimido,
│                                        1 133 238 x 1088) + .pesos.txt que dice con qué pesos se sacó cada uno
├── m4/salida/capilla_etiquetada/        grupos (m4_original, m4_adaptado) y evaluaciones con el M2 de SemanticPOSS
├── arch/                                ArCH (dataset público de patrimonio, CC BY-NC-SA 4.0)
│   ├── datos/ArCH_labelled/{Training,Test}/   17 .txt originales (~10 GB), bajados por FTP desde Colab
│   ├── resumen/                         resumen_arch.md/json: puntos y clases por escena
│   ├── tiles/                           bloques de 10 m para entrenar (los crea colab_entrenamiento_arch.ipynb)
│   └── entrenamiento/                   M2 entrenado con ArCH (corrida del 10-nov-2026)
│       ├── checkpoints/ArCH/corrida_completa/   ArCH_S0_best-epoch=20-step=860.ckpt (mejor) y last.ckpt (época 30), ~450 MB c/u
│       ├── logs/                        metrics.csv de Lightning (version_3 = la corrida completa)
│       ├── medicion/                    tiempo, RAM y GPU de cada paso (conversión, prueba, corrida, predicciones)
│       ├── predicciones/                f_sem de la capilla con M2-ArCH: capilla_etiquetada_{best,last}_fsem.npz (~150 MB c/u)
│       └── (m4_capilla_arch_e30/)       (vacía: M4 con M2-ArCH se corrió en la laptop; sus grupos están en
│                                        experimentacion/arch_entrenamiento/resultados/m4_capilla/m4_*/ — súbelos aquí si quieres respaldo)
└── entornos/                            entornos compilados para no reinstalar
    ├── pointcept_env_py<...>_torch<...>_cu<...>.tar.gz      (Sonata/Pointcept en Colab)
    └── hyperncd_env_cache.tar.gz                            (HyperNCD / MinkowskiEngine en Colab)
```

Notas:
- Los entornos compilados dependen de la GPU de la sesión: si cambias de tipo de GPU, hay que
  recompilar las extensiones (`torch-scatter`, `torch-cluster`, `pointops`).
- Si ves una carpeta `model_prueba_10it` dentro de `m3/entrenamiento/`, es la corrida inválida
  (10 iteraciones por época, por un error de `loop`). No sirve para comparar y se puede borrar.
- Fuera de esta carpeta hay material de la asesoría (papers, `Otros.txt`, propuesta) en
  "Manyory C./ Renzo F." compartido por el asesor; no lo toca el pipeline.
- Nada de Drive va a git. La copia local de lo necesario está en `experimentacion/datos/` y `cache/`.

## Dónde está cada cosa, por experimento

Regla: lo **liviano** (README, `.json`, `.csv`, `.png`) va a git; lo **pesado** (`.npz`, `.npy`, `.ckpt`, `.pth`, `.html`) vive en Drive y,
si hizo falta, en una copia local que **no** va a git. Cada experimento anota sus rutas pesadas en su registro
(`experimentacion/registro/<id>.json`, campo `datos_pesados`) y en su README.

| Experimento | En git (liviano) | En Drive (pesado) | Copia local, no va a git |
|---|---|---|---|
| M2 entrenado con ArCH (`m2_arch_entrenamiento`) | `experimentacion/arch_entrenamiento/` (README, `resultados/entrenamiento/`) | `arch/entrenamiento/checkpoints,logs,medicion` y `arch/tiles/` | `arch_entrenamiento/resultados_colab/` (el zip de Colab) |
| M2-ArCH en la capilla (`escena_m2_arch_e30`) | `arch_entrenamiento/resultados/m2_capilla/` | `arch/entrenamiento/predicciones/*.npz` | `arch_entrenamiento/resultados_colab/predicciones/` |
| M4 con M2-ArCH en la capilla | `arch_entrenamiento/resultados/m4_capilla/` (tablas e imágenes) | `arch/entrenamiento/m4_capilla_*` | `m4_capilla/m4_{original,adaptado}/grupos_PL13.npz` |
| Escena etiquetada con M2 de SemanticPOSS (`escena_*`) | `experimentacion/escena_capilla_etiquetada/` (README, `resultados/`) | `m1..m4/salida/capilla_etiquetada/` | `escena_capilla_etiquetada/datos/` (incluye los `f_ssl`) |
| M2 colapso por dominio | `experimentacion/m2_dominio/` | los `f_sem` de PL7/PL13 en `m2/salida/resultados_m2/` | `cache/` |
| Adaptación de Sonata (M3) | `experimentacion/adaptacion_sonata/` | `m3/entrenamiento/`, `m3/pesos/` | `experimentacion/datos/modulo3_*` |
| ArCH (datos) | `experimentacion/arch_resumen/` | `arch/datos/`, `arch/resumen/` | — |

Si un compañero necesita rehacer algo: (1) abre el `README` del experimento, (2) mira su registro en `experimentacion/registro/` (qué entradas usó),
(3) baja de Drive solo la ruta de `datos_pesados` que necesite. Índice de experimentos y visor: `experimentacion/registro/INDICE.md` y `experimentacion/visor/index.html`.
